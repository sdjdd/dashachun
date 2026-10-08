use std::sync::Arc;

use sqlx::PgPool;

use super::memory::MemoryFactory;
use super::store::AgentStore;
use super::transcript::{DbTranscriptSink, TranscriptOwner};
use super::{Agent, CompositeAgent, SystemPrompt, ToolRegistry};
use crate::asr::Asr;
use crate::device::DeviceRecord;
use crate::error::AppError;
use crate::llm::Llm;
use crate::tts::Tts;
use crate::vad::VadFactory;

/// Shared, expensive provider bundle plus the per-device configuration source.
/// One factory lives for the process lifetime; it builds a fresh, cheap
/// [`CompositeAgent`] per connection with the device's persona prompt and a
/// transcript sink attributed to the device.
pub struct AgentFactory {
    asr: Arc<dyn Asr>,
    llm: Arc<dyn Llm>,
    tts: Arc<dyn Tts>,
    vad: Arc<dyn VadFactory>,
    memory: Arc<dyn MemoryFactory>,
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
        memory: Arc<dyn MemoryFactory>,
        tools: Arc<ToolRegistry>,
        agents: AgentStore,
        pool: PgPool,
    ) -> Self {
        Self {
            asr,
            llm,
            tts,
            vad,
            memory,
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
        let transcript = DbTranscriptSink::new(
            self.pool.clone(),
            TranscriptOwner {
                user_id,
                agent_id,
                client_id: record.client_id,
                device_id: record.device_id.clone(),
            },
        );
        Ok(Arc::new(CompositeAgent {
            asr: self.asr.clone(),
            llm: self.llm.clone(),
            tts: self.tts.clone(),
            vad: self.vad.clone(),
            memory: self.memory.build(),
            tools: self.tools.clone(),
            system_prompt: SystemPrompt::new(&agent.persona_prompt),
            transcript: Some(Arc::new(transcript)),
        }))
    }
}
