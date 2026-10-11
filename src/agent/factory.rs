use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;

use super::capture::{AudioCapture, AudioStorage};
use super::memory::{DbEntryMemory, DbMemory, EntryMemory, MemoryHook, MemoryOwner};
use super::store::AgentStore;
use super::tool::{MemoryAdd, MemoryDelete, MemoryUpdate};
use super::{Agent, Asr, Capture, CompositeAgent, Llm, Memory, SystemPrompt, ToolEntry, Tts};
use crate::device::DeviceRecord;
use crate::error::AppError;
use crate::provider::{ProviderRegistry, ProviderSpec};
use crate::vad::VadFactory;

/// One generation of the shared provider bundle plus the per-device
/// configuration source. Built through the [`ProviderRegistry`] from three
/// [`ProviderSpec`]s via [`AgentFactory::assemble`]; [`crate::state::ServerState`]
/// publishes the current generation behind an `ArcSwap`, so a config change
/// assembles a whole new generation and swaps it in — in-flight sessions keep
/// their provider clones, and an orphaned generation drains its TTS pool
/// through [`AgentFactory::drain_when_orphaned`].
pub struct AgentFactory {
    asr: Arc<dyn Asr>,
    llm: Arc<dyn Llm>,
    tts: Arc<dyn Tts>,
    vad: Arc<dyn VadFactory>,
    tools: Vec<ToolEntry>,
    agents: AgentStore,
    pool: PgPool,
}

impl AgentFactory {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        asr: Arc<dyn Asr>,
        llm: Arc<dyn Llm>,
        tts: Arc<dyn Tts>,
        vad: Arc<dyn VadFactory>,
        tools: Vec<ToolEntry>,
        agents: AgentStore,
        pool: PgPool,
    ) -> Self {
        Self {
            asr,
            llm,
            tts,
            vad,
            tools,
            agents,
            pool,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        registry: &ProviderRegistry,
        asr: &ProviderSpec,
        llm: &ProviderSpec,
        tts: &ProviderSpec,
        vad: Arc<dyn VadFactory>,
        tools: Vec<ToolEntry>,
        agents: AgentStore,
        pool: PgPool,
    ) -> Result<Self, String> {
        tracing::info!(stage = "asr", kind = %asr.kind, "constructing provider");
        let asr = registry.build_asr(&asr.kind, &asr.config)?;
        tracing::info!(
            stage = "llm",
            kind = %llm.kind,
            model = json_str(&llm.config, "model"),
            "constructing provider"
        );
        let llm = registry.build_llm(&llm.kind, &llm.config)?;
        tracing::info!(stage = "tts", kind = %tts.kind, "constructing provider");
        let tts = registry.build_tts(&tts.kind, &tts.config)?;
        Ok(Self::new(asr, llm, tts, vad, tools, agents, pool))
    }

    pub async fn build(
        &self,
        record: &DeviceRecord,
        audio_storage: Option<Arc<dyn AudioStorage>>,
    ) -> Result<Arc<dyn Agent>, AppError> {
        let agent_id = record
            .agent_id
            .ok_or_else(|| AppError::Internal("device has no assigned agent".into()))?;
        let user_id = record
            .user_id
            .ok_or_else(|| AppError::Internal("device is not bound".into()))?;
        let agent = self
            .agents
            .find(agent_id)
            .await?
            .ok_or_else(|| AppError::Internal(format!("agent {agent_id} not found")))?;
        let owner = MemoryOwner {
            user_id,
            agent_id,
            client_id: record.client_id,
            device_id: record.device_id.clone(),
        };
        // The audio feature plugs in here: it implements both the driver's
        // Capture surface and the memory's MemoryHook, so committed rows
        // hand their ids straight to the open captures. The storage is the
        // caller's slot snapshot, taken per connection — a settings swap
        // applies to every new session without touching the generation.
        let audio = Arc::new(AudioCapture::new(
            self.pool.clone(),
            record.client_id,
            audio_storage,
        ));
        let memory = DbMemory::load(
            self.pool.clone(),
            owner,
            vec![audio.clone() as Arc<dyn MemoryHook>],
        )
        .await?;
        // The entry memory is scope-bound per connection; the memory tools
        // hold the same instance, so the prompt reads what they write. The
        // base registry is generation-shared — each connection extends a
        // clone of it with its own memory tools.
        let entries: Arc<dyn EntryMemory> =
            Arc::new(DbEntryMemory::new(self.pool.clone(), user_id, agent_id));
        let mut tools = self.tools.clone();
        tools.push(ToolEntry::new(MemoryAdd::new(entries.clone())));
        tools.push(ToolEntry::new(MemoryUpdate::new(entries.clone())));
        tools.push(ToolEntry::new(MemoryDelete::new(entries.clone())));
        Ok(Arc::new(CompositeAgent {
            asr: self.asr.clone(),
            llm: self.llm.clone(),
            tts: self.tts.clone(),
            vad: self.vad.clone(),
            memory: Arc::new(memory) as Arc<dyn Memory>,
            entries,
            tools,
            system_prompt: SystemPrompt::new(&agent.persona_prompt),
            capture: Some(audio as Arc<dyn Capture>),
        }))
    }

    pub async fn shutdown(&self) {
        self.tts.shutdown().await;
    }

    /// Drains this generation's TTS pool once no session or factory snapshot
    /// still holds its providers — the removal path for a swapped-out
    /// generation.
    pub fn drain_when_orphaned(self: Arc<Self>) {
        let tts = self.tts.clone();
        drop(self);
        tokio::spawn(async move {
            while Arc::strong_count(&tts) > 1 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            tts.shutdown().await;
        });
    }
}

fn json_str<'a>(config: &'a serde_json::Value, field: &str) -> &'a str {
    config
        .get(field)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?")
}
