mod listen;
mod reply;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::agent::{
    Agent, AgentInputStream, AgentOutput, AgentOutputStream, AgentSession, Asr, Capture,
    EntryMemory, Llm, Memory, Speaker, SystemPrompt, ToolSet, Tts,
};
use crate::util::recv_stream;
use crate::vad::VadFactory;
use crate::voiceprint::SpeakerMatcher;

use listen::listen_stage;
use reply::{ReplyEvent, reply_stage};

const ASR_CHANNEL_CAPACITY: usize = 64;
const UTTERANCE_CHANNEL_CAPACITY: usize = 64;
const TURN_EVENT_CAPACITY: usize = 64;
const TTS_TEXT_CHANNEL_CAPACITY: usize = 64;
const OUTPUT_CHANNEL_CAPACITY: usize = 64;
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct CompositeAgent {
    pub asr: Arc<dyn Asr>,
    pub llm: Arc<dyn Llm>,
    pub tts: Arc<dyn Tts>,
    pub vad: Arc<dyn VadFactory>,
    pub memory: Arc<dyn Memory>,
    pub entries: Arc<dyn EntryMemory>,
    pub tools: Arc<ToolSet>,
    pub system_prompt: SystemPrompt,
    pub capture: Option<Arc<dyn Capture>>,
    pub speaker_matcher: Option<Arc<SpeakerMatcher>>,
    /// The roster behind the matcher: the agent's enrolled speakers as
    /// they were when the device connected, as prompt data. Both derive
    /// from the same factory snapshot — a roster entry exists exactly when
    /// the matcher can identify it.
    pub speakers: Vec<Speaker>,
}

impl Agent for CompositeAgent {
    fn run(&self, session: AgentSession, input: AgentInputStream) -> AgentOutputStream {
        let (out_tx, out_rx) = mpsc::channel::<AgentOutput>(OUTPUT_CHANNEL_CAPACITY);
        let session_id = session.id.clone();
        info!(session_id, "agent started");
        tokio::spawn({
            let agent = self.clone();
            async move {
                drive(agent, session, input, out_tx).await;
                info!(session_id, "agent stopped");
            }
        });
        Box::pin(recv_stream(out_rx))
    }
}

/// What the listening side hands to the reply side. Each utterance carries
/// its own cancellation token: the listener keeps the authority to kill the
/// in-flight reply the moment new speech or an interrupt arrives, the reply
/// side only observes it. The speaker is the matched speaker's voiceprint
/// row id — `None` when this utterance identified no one (below threshold,
/// failed, or not ready in time); the reply stage labels the context's
/// user message with it and never sees a voiceprint type.
enum ListenEvent {
    Utterance {
        text: String,
        speaker: Option<i64>,
        turn: CancellationToken,
    },
    Interrupt,
    Reset,
}

async fn drive(
    agent: CompositeAgent,
    session: AgentSession,
    input: AgentInputStream,
    out_tx: mpsc::Sender<AgentOutput>,
) {
    let (utterances, utterance_rx) = mpsc::channel::<ListenEvent>(UTTERANCE_CHANNEL_CAPACITY);
    let (events_tx, events_rx) = mpsc::channel::<(u64, ReplyEvent)>(TURN_EVENT_CAPACITY);
    tokio::spawn(listen_stage(
        agent.clone(),
        session.clone(),
        input,
        utterances,
        out_tx.clone(),
    ));
    reply_stage(agent, session, utterance_rx, events_tx, events_rx, out_tx).await;
}

/// A spawned task a cut can stop politely: `cancel` fires the token and
/// hands the join handle to a detached waiter that gives the task
/// `CANCEL_TIMEOUT` to exit before aborting; dropping the task aborts the
/// handle immediately; `detach` surrenders the handle so the task runs to
/// completion on its own.
struct Cancellable {
    token: CancellationToken,
    handle: Option<JoinHandle<()>>,
}

impl Cancellable {
    fn new(token: CancellationToken, handle: JoinHandle<()>) -> Self {
        Self {
            token,
            handle: Some(handle),
        }
    }

    fn cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    fn detach(&mut self) {
        self.handle.take();
    }

    fn cancel(mut self) {
        self.token.cancel();
        if let Some(mut handle) = self.handle.take() {
            tokio::spawn(async move {
                stop_task(&mut handle).await;
            });
        }
    }
}

impl Drop for Cancellable {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn stop_task(handle: &mut JoinHandle<()>) {
    if tokio::time::timeout(CANCEL_TIMEOUT, &mut *handle)
        .await
        .is_err()
    {
        warn!("provider did not cancel in time, aborting");
        handle.abort();
    }
}
