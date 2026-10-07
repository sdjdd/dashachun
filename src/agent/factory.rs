use std::sync::Arc;

use super::memory::MemoryFactory;
use super::store::AgentStore;
use super::{Agent, CompositeAgent, SystemPrompt, ToolRegistry};
use crate::asr::Asr;
use crate::error::AppError;
use crate::llm::Llm;
use crate::tts::Tts;
use crate::vad::VadFactory;

/// Shared, expensive provider bundle plus the per-device configuration source.
/// One factory lives for the process lifetime; it builds a fresh, cheap
/// [`CompositeAgent`] per connection with the device's persona prompt.
pub struct AgentFactory {
    asr: Arc<dyn Asr>,
    llm: Arc<dyn Llm>,
    tts: Arc<dyn Tts>,
    vad: Arc<dyn VadFactory>,
    memory: Arc<dyn MemoryFactory>,
    tools: Arc<ToolRegistry>,
    agents: AgentStore,
}

impl AgentFactory {
    pub fn new(
        asr: Arc<dyn Asr>,
        llm: Arc<dyn Llm>,
        tts: Arc<dyn Tts>,
        vad: Arc<dyn VadFactory>,
        memory: Arc<dyn MemoryFactory>,
        tools: Arc<ToolRegistry>,
        agents: AgentStore,
    ) -> Self {
        Self {
            asr,
            llm,
            tts,
            vad,
            memory,
            tools,
            agents,
        }
    }

    pub async fn build(&self, agent_id: Option<i64>) -> Result<Arc<dyn Agent>, AppError> {
        let agent_id =
            agent_id.ok_or_else(|| AppError::Internal("device has no assigned agent".into()))?;
        let record = self
            .agents
            .find(agent_id)
            .await?
            .ok_or_else(|| AppError::Internal(format!("agent {agent_id} not found")))?;
        Ok(Arc::new(CompositeAgent {
            asr: self.asr.clone(),
            llm: self.llm.clone(),
            tts: self.tts.clone(),
            vad: self.vad.clone(),
            memory: self.memory.build(),
            tools: self.tools.clone(),
            system_prompt: SystemPrompt::new(&record.persona_prompt),
        }))
    }
}
