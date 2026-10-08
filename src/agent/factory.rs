use std::sync::Arc;

use sqlx::PgPool;

use super::capture::AudioCapture;
use super::memory::{DbMemory, MemoryHook, MemoryOwner};
use super::store::AgentStore;
use super::{Agent, Capture, CompositeAgent, Memory, SystemPrompt, ToolRegistry};
use crate::asr::Asr;
use crate::device::DeviceRecord;
use crate::error::AppError;
use crate::llm::Llm;
use crate::tts::Tts;
use crate::vad::VadFactory;

/// Shared, expensive provider bundle plus the per-device configuration source.
/// One factory lives for the process lifetime; it builds a fresh, cheap
/// [`CompositeAgent`] per connection with the device's persona prompt, a
/// transcript sink attributed to the device, and the conversation memory
/// preloaded from the same device's transcript rows.
pub struct AgentFactory {
    asr: Arc<dyn Asr>,
    llm: Arc<dyn Llm>,
    tts: Arc<dyn Tts>,
    vad: Arc<dyn VadFactory>,
    tools: Arc<ToolRegistry>,
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
        tools: Arc<ToolRegistry>,
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

    pub async fn build(&self, record: &DeviceRecord) -> Result<Arc<dyn Agent>, AppError> {
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
        // hand their ids straight to the open captures.
        let audio = Arc::new(AudioCapture::new(self.pool.clone()));
        let memory = DbMemory::load(
            self.pool.clone(),
            owner,
            vec![audio.clone() as Arc<dyn MemoryHook>],
        )
        .await?;
        Ok(Arc::new(CompositeAgent {
            asr: self.asr.clone(),
            llm: self.llm.clone(),
            tts: self.tts.clone(),
            vad: self.vad.clone(),
            memory: Arc::new(memory) as Arc<dyn Memory>,
            tools: self.tools.clone(),
            system_prompt: SystemPrompt::new(&agent.persona_prompt),
            capture: Some(audio as Arc<dyn Capture>),
        }))
    }
}
