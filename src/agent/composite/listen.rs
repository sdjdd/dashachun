//! The listening leg: the device input, the VAD and the per-utterance ASR
//! tasks for the whole session. Forwards `Stt`/`Mcp`/`Error` directly and
//! hands every final utterance to the reply stage together with a fresh
//! turn token, cancelling the previous one first.

use std::sync::Arc;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::{ASR_CHANNEL_CAPACITY, Cancellable, CompositeAgent, ListenEvent};
use crate::agent::{
    AgentInput, AgentInputStream, AgentOutput, AgentSession, Asr, AsrEvent, AudioStream, Capture,
    Memory, UtteranceCapture,
};
use crate::util::recv_stream;
use crate::vad::{Vad, VadEvent};

enum AsrUpdate {
    Partial { epoch: u64, text: String },
    Final { epoch: u64, text: String },
    Error { epoch: u64, message: String },
}

pub(super) async fn listen_stage(
    agent: CompositeAgent,
    session: AgentSession,
    mut input: AgentInputStream,
    utterances: mpsc::Sender<ListenEvent>,
    out_tx: mpsc::Sender<AgentOutput>,
) {
    let mut vad: Option<Box<dyn Vad>> = None;
    let (asr_tx, mut asr_rx) = mpsc::channel::<AsrUpdate>(ASR_CHANNEL_CAPACITY);
    let mut slot: Option<AsrSlot> = None;
    let mut epoch: u64 = 0;
    let mut live_turn: Option<CancellationToken> = None;
    loop {
        tokio::select! {
            item = input.next() => {
                let Some(item) = item else { break };
                match item {
                    AgentInput::ListenStart { .. } => {
                        epoch += 1;
                        if let Some(current) = slot.take() {
                            current.cancel();
                        }
                        if let Some(turn) = live_turn.take() {
                            turn.cancel();
                        }
                        if utterances.send(ListenEvent::Reset).await.is_err() {
                            break;
                        }
                        vad = match agent.vad.build(session.sample_rate) {
                            Ok(vad) => Some(vad),
                            Err(err) => {
                                warn!(session_id = %session.id, %err, "failed to build vad");
                                None
                            }
                        };
                    }
                    AgentInput::ListenStop => {
                        info!(session_id = %session.id, "listen stop");
                        if let Some(vad) = vad.as_mut() {
                            let events = vad.flush();
                            handle_vad_events(
                                events,
                                listen_deps(&session, &agent, &asr_tx),
                                &mut epoch,
                                &mut slot,
                            )
                            .await;
                        }
                        slot = None;
                    }
                    AgentInput::Audio(samples) => {
                        let Some(vad) = vad.as_mut() else { continue };
                        let events = vad.push(&samples);
                        handle_vad_events(
                            events,
                            listen_deps(&session, &agent, &asr_tx),
                            &mut epoch,
                            &mut slot,
                        )
                        .await;
                    }
                    AgentInput::Interrupt { reason } => {
                        info!(session_id = %session.id, reason = ?reason, "interrupt");
                        epoch += 1;
                        if let Some(current) = slot.take() {
                            current.cancel();
                        }
                        if let Some(turn) = live_turn.take() {
                            turn.cancel();
                        }
                        if utterances.send(ListenEvent::Interrupt).await.is_err() {
                            break;
                        }
                    }
                    AgentInput::Mcp(payload) => {
                        if out_tx.send(AgentOutput::Mcp(payload)).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Some(message) = asr_rx.recv() => {
                let current = match &message {
                    AsrUpdate::Partial { epoch, .. }
                    | AsrUpdate::Final { epoch, .. }
                    | AsrUpdate::Error { epoch, .. } => *epoch,
                };
                if current != epoch {
                    continue;
                }
                match message {
                    AsrUpdate::Partial { text, .. } => {
                        if out_tx
                            .send(AgentOutput::Stt {
                                text,
                                is_final: false,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    AsrUpdate::Final { text, .. } => {
                        info!(session_id = %session.id, %text, "asr result");
                        if out_tx
                            .send(AgentOutput::Stt {
                                text: text.clone(),
                                is_final: true,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                        if text.is_empty() {
                            continue;
                        }
                        let turn = CancellationToken::new();
                        if let Some(previous) = live_turn.replace(turn.clone()) {
                            previous.cancel();
                        }
                        if utterances
                            .send(ListenEvent::Utterance { text, turn })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    AsrUpdate::Error { message, .. } => {
                        warn!(session_id = %session.id, %message, "asr failed");
                        if out_tx.send(AgentOutput::Error { message }).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }
    if let Some(current) = slot.take() {
        current.cancel();
    }
    if let Some(turn) = live_turn.take() {
        turn.cancel();
    }
}

/// What the VAD-event handling needs: the shared stage inputs.
fn listen_deps<'a>(
    session: &'a AgentSession,
    agent: &'a CompositeAgent,
    asr_tx: &'a mpsc::Sender<AsrUpdate>,
) -> ListenDeps<'a> {
    ListenDeps {
        session,
        asr: &agent.asr,
        asr_tx,
        memory: &agent.memory,
        capture: agent.capture.as_deref(),
    }
}

struct ListenDeps<'a> {
    session: &'a AgentSession,
    asr: &'a Arc<dyn Asr>,
    asr_tx: &'a mpsc::Sender<AsrUpdate>,
    memory: &'a Arc<dyn Memory>,
    capture: Option<&'a dyn Capture>,
}

async fn handle_vad_events(
    events: Vec<VadEvent>,
    deps: ListenDeps<'_>,
    epoch: &mut u64,
    slot: &mut Option<AsrSlot>,
) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id = %deps.session.id, at_ms, "speech start");
                *epoch += 1;
                let capture = deps
                    .capture
                    .map(|capture| capture.start_utterance(deps.session));
                *slot = Some(spawn_asr(&deps, *epoch, capture));
            }
            VadEvent::Speech { samples } => {
                if let Some(current) = slot.as_ref() {
                    if let Some(capture) = &current.capture {
                        capture.push(&samples);
                    }
                    if current.tx.send(samples).await.is_err() {
                        debug!(session_id = %deps.session.id, "asr input closed, dropping frame");
                    }
                }
            }
            VadEvent::SpeechEnd { .. } => {
                if let Some(current) = slot.take() {
                    current.detach();
                }
            }
        }
    }
}

/// One live per-utterance ASR task: its audio feed, the cancellable task
/// and the utterance capture.
struct AsrSlot {
    tx: mpsc::Sender<Vec<f32>>,
    task: Cancellable,
    capture: Option<UtteranceCapture>,
}

impl AsrSlot {
    /// Lets the task run to its final on its own.
    fn detach(mut self) {
        self.task.detach();
    }

    fn cancel(self) {
        self.task.cancel();
    }
}

fn spawn_asr(deps: &ListenDeps<'_>, epoch: u64, capture: Option<UtteranceCapture>) -> AsrSlot {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(ASR_CHANNEL_CAPACITY);
    let audio: AudioStream = recv_stream(rx);
    let cancel = CancellationToken::new();
    let handle = tokio::spawn(run_asr(
        deps.asr.clone(),
        audio,
        deps.asr_tx.clone(),
        deps.session.clone(),
        epoch,
        cancel.clone(),
        deps.memory.clone(),
        capture.clone(),
    ));
    AsrSlot {
        tx,
        task: Cancellable::new(cancel, handle),
        capture,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_asr(
    asr: Arc<dyn Asr>,
    audio: AudioStream,
    asr_tx: mpsc::Sender<AsrUpdate>,
    session: AgentSession,
    epoch: u64,
    cancel: CancellationToken,
    memory: Arc<dyn Memory>,
    capture: Option<UtteranceCapture>,
) {
    let mut events = asr.transcribe(audio, cancel.clone());
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(AsrEvent::Partial { text }) => AsrUpdate::Partial { epoch, text },
            Ok(AsrEvent::Final { text }) => {
                // The utterance is real speech even when the turn was
                // superseded, so the row is stored regardless of the cancel
                // token; the capture learns its message id through the
                // memory's hook, not from here.
                if !cancel.is_cancelled() && !text.is_empty() {
                    match memory.store_utterance(&session, &text).await {
                        Ok(()) => {
                            if let Some(capture) = capture {
                                capture.finish();
                            }
                        }
                        Err(err) => {
                            warn!(session_id = %session.id, %err, "failed to store utterance");
                        }
                    }
                }
                let _ = asr_tx.send(AsrUpdate::Final { epoch, text }).await;
                return;
            }
            Err(err) => AsrUpdate::Error {
                epoch,
                message: err.to_string(),
            },
        };
        if asr_tx.send(message).await.is_err() {
            debug!(session_id = %session.id, "asr output closed");
            return;
        }
    }
}
