use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;
use tokio::sync::Mutex;

use super::capture::{AudioCapture, AudioStorage};
use super::memory::{DbEntryMemory, DbMemory, EntryMemory, Extractor, MemoryHook};
use super::store::AgentStore;
use super::tool::memory_tools;
use super::{
    Agent, Asr, Capture, CompositeAgent, Llm, Memory, Speaker, SystemPrompt, ToolSet, Tts,
};
use crate::device::DeviceRecord;
use crate::error::AppError;
use crate::provider::{ProviderRegistry, ProviderSpec};
use crate::vad::VadFactory;
use crate::voiceprint::{SpeakerMatcher, SpeakerStore, Voiceprint};

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
    voiceprint: Option<Arc<Mutex<Voiceprint>>>,
    tools: ToolSet,
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
        voiceprint: Option<Arc<Mutex<Voiceprint>>>,
        tools: ToolSet,
        agents: AgentStore,
        pool: PgPool,
    ) -> Self {
        Self {
            asr,
            llm,
            tts,
            vad,
            voiceprint,
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
        voiceprint: Option<Arc<Mutex<Voiceprint>>>,
        tools: ToolSet,
        agents: AgentStore,
        pool: PgPool,
    ) -> Result<Self, String> {
        tracing::info!(stage = "asr", kind = %asr.kind, "constructing provider");
        let asr = registry.build_asr(asr)?;
        tracing::info!(
            stage = "llm",
            kind = %llm.kind,
            model = json_str(&llm.config, "model"),
            "constructing provider"
        );
        let llm = registry.build_llm(llm)?;
        tracing::info!(stage = "tts", kind = %tts.kind, "constructing provider");
        let tts = registry.build_tts(tts)?;
        Ok(Self::new(
            asr, llm, tts, vad, voiceprint, tools, agents, pool,
        ))
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
        // The entry memory is scope-bound per connection; the memory tools
        // hold the same instance, so the prompt reads what they write. The
        // base registry is generation-shared — each connection extends a
        // clone of it with its own memory tools.
        let entries: Arc<dyn EntryMemory> =
            Arc::new(DbEntryMemory::new(self.pool.clone(), user_id, agent_id));
        // The audio feature plugs in here: the driver's Capture surface —
        // each capture handle carries its own id channel, and the store
        // call sites bind the committed row's id to the very capture that
        // produced the audio. The storage is the caller's slot snapshot,
        // taken per connection — a settings swap applies to every new
        // session without touching the generation.
        let audio = Arc::new(AudioCapture::new(
            self.pool.clone(),
            record.client_id,
            audio_storage,
        ));
        // The voiceprint feature plugs in here like the audio capture: the
        // model is the boot slot, the agent's enrolled speakers are read
        // once per session — as the matcher's candidates and, from the same
        // snapshot, as the prompt's known-speakers roster (a roster entry
        // exists exactly when the matcher can identify it). An agent
        // without enrollments — or without the model — gets neither and
        // runs no identification pipeline at all; enrollments made later
        // join at the next session.
        let (speaker_matcher, speakers) = match &self.voiceprint {
            Some(model) => {
                let enrolled = SpeakerStore::new(self.pool.clone())
                    .embeddings_for_agent(agent_id)
                    .await?;
                let speakers = enrolled
                    .iter()
                    .map(|speaker| Speaker {
                        id: speaker.id,
                        name: speaker.name.clone(),
                        description: speaker.description.clone(),
                    })
                    .collect();
                if !enrolled.is_empty() {
                    tracing::info!(
                        agent_id,
                        speakers = enrolled.len(),
                        "voiceprint identification enabled"
                    );
                }
                (
                    SpeakerMatcher::new(model.clone(), enrolled).map(Arc::new),
                    speakers,
                )
            }
            None => (None, Vec::new()),
        };
        // The background extractor is the memory's one hook: it sees
        // exactly this connection's window appends — the announced batches
        // buffer until a batch is full, then one background extraction
        // proposes entry changes against the same entry memory the tools
        // write and the prompt reads, with the session's speaker roster in
        // its prompt.
        let extractor = Arc::new(Extractor::new(
            Arc::clone(&self.llm),
            Arc::clone(&entries),
            speakers.clone(),
        ));
        let memory = DbMemory::new(self.pool.clone(), vec![extractor as Arc<dyn MemoryHook>]);
        let tools = self.tools.clone().merge(memory_tools(&entries));
        Ok(Arc::new(CompositeAgent {
            asr: self.asr.clone(),
            llm: self.llm.clone(),
            tts: self.tts.clone(),
            vad: self.vad.clone(),
            memory: Arc::new(memory) as Arc<dyn Memory>,
            entries,
            tools: Arc::new(tools),
            system_prompt: SystemPrompt::new(&agent.persona_prompt),
            capture: Some(audio as Arc<dyn Capture>),
            speaker_matcher,
            speakers,
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
