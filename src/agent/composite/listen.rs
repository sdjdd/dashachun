//! The listening leg: the device input, the VAD and the per-utterance ASR
//! tasks for the whole session. Forwards `Stt`/`Mcp`/`Error` directly and
//! hands every final utterance to the reply stage together with a fresh
//! turn token and the utterance's voiceprint attribution, cancelling the
//! previous one first.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::{ASR_CHANNEL_CAPACITY, Cancellable, CompositeAgent, ListenEvent};
use crate::agent::{
    AgentInput, AgentInputStream, AgentOutput, AgentSession, Asr, AsrEvent, AudioStream, Capture,
    Memory, Speaker, UtteranceCapture,
};
use crate::util::recv_stream;
use crate::vad::{Vad, VadEvent};
use crate::voiceprint::{Identification, SAMPLE_RATE, SpeakerMatcher, SpeakerTap};

/// How long the ASR final waits for the tap's identification before the
/// turn starts unlabeled. The tap finishes at `SpeechEnd` and the embed
/// takes tens of milliseconds, so the result is usually already resolved
/// when the final arrives — the wait only bounds a rare tail (a long
/// segment, the model mutex held by an enrollment); an expiry starts the
/// turn without attribution, which still logs.
const SPEAKER_WAIT: Duration = Duration::from_millis(250);

/// The identification outcome of the utterance between `SpeechEnd` and its
/// ASR final.
type PendingSpeaker = oneshot::Receiver<Option<Identification>>;

/// What the prompt renders for an utterance that identified no one.
const UNKNOWN: &str = "未知";

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
    let mut pending_speaker: Option<PendingSpeaker> = None;
    // The identification pipeline is a per-session capability: the matcher
    // exists only when the agent has enrolled voiceprints and the model
    // loaded, and the frontend only speaks 16 kHz — anything else runs
    // without it (the repo has no resampler).
    let speaker = match agent.speaker_matcher.as_ref() {
        Some(matcher) if session.sample_rate == SAMPLE_RATE => Some(matcher),
        Some(_) => {
            debug!(
                session_id = %session.id,
                "voiceprint identification skipped: the uplink is not 16 kHz"
            );
            None
        }
        None => None,
    };
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
                        pending_speaker = None;
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
                                listen_deps(&session, &agent, &asr_tx, speaker),
                                &mut epoch,
                                &mut slot,
                                &mut pending_speaker,
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
                            listen_deps(&session, &agent, &asr_tx, speaker),
                            &mut epoch,
                            &mut slot,
                            &mut pending_speaker,
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
                        pending_speaker = None;
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
                        let speaker =
                            resolve_speaker(pending_speaker.take(), speaker.is_some()).await;
                        let turn = CancellationToken::new();
                        if let Some(previous) = live_turn.replace(turn.clone()) {
                            previous.cancel();
                        }
                        if utterances
                            .send(ListenEvent::Utterance {
                                text,
                                speaker,
                                turn,
                            })
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
    speaker: Option<&'a Arc<SpeakerMatcher>>,
) -> ListenDeps<'a> {
    ListenDeps {
        session,
        asr: &agent.asr,
        asr_tx,
        memory: &agent.memory,
        capture: agent.capture.as_deref(),
        speaker,
    }
}

struct ListenDeps<'a> {
    session: &'a AgentSession,
    asr: &'a Arc<dyn Asr>,
    asr_tx: &'a mpsc::Sender<AsrUpdate>,
    memory: &'a Arc<dyn Memory>,
    capture: Option<&'a dyn Capture>,
    speaker: Option<&'a Arc<SpeakerMatcher>>,
}

async fn handle_vad_events(
    events: Vec<VadEvent>,
    deps: ListenDeps<'_>,
    epoch: &mut u64,
    slot: &mut Option<AsrSlot>,
    pending_speaker: &mut Option<PendingSpeaker>,
) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id = %deps.session.id, at_ms, "speech start");
                *epoch += 1;
                // The previous utterance is superseded: whatever its tap
                // produced is no longer reachable by an ASR final.
                *pending_speaker = None;
                let capture = deps
                    .capture
                    .map(|capture| capture.start_utterance(deps.session));
                let tap = deps.speaker.map(|speaker| speaker.tap(deps.session));
                *slot = Some(spawn_asr(&deps, *epoch, capture, tap));
            }
            VadEvent::Speech { samples } => {
                if let Some(current) = slot.as_ref() {
                    if let Some(capture) = &current.capture {
                        capture.push(&samples);
                    }
                    if let Some(tap) = &current.tap {
                        tap.push(&samples);
                    }
                    if current.tx.send(samples).await.is_err() {
                        debug!(session_id = %deps.session.id, "asr input closed, dropping frame");
                    }
                }
            }
            VadEvent::SpeechEnd { .. } => {
                if let Some(current) = slot.take() {
                    *pending_speaker = current.detach();
                }
            }
        }
    }
}

/// One live per-utterance ASR task: its audio feed, the cancellable task,
/// the utterance capture and the voiceprint tap.
struct AsrSlot {
    tx: mpsc::Sender<Vec<f32>>,
    task: Cancellable,
    capture: Option<UtteranceCapture>,
    tap: Option<SpeakerTap>,
}

impl AsrSlot {
    /// Lets the task run to its final on its own and closes the utterance:
    /// the capture finishes inside the task (it learns its message id
    /// there), the voiceprint tap finishes here — the speech segment is
    /// complete, identification does not wait for the ASR final — and the
    /// tap's result channel returns to the stage, which resolves it when
    /// this utterance's final turns into a turn.
    fn detach(mut self) -> Option<PendingSpeaker> {
        self.task.detach();
        self.tap.take().map(SpeakerTap::finish)
    }

    fn cancel(self) {
        self.task.cancel();
    }
}

fn spawn_asr(
    deps: &ListenDeps<'_>,
    epoch: u64,
    capture: Option<UtteranceCapture>,
    tap: Option<SpeakerTap>,
) -> AsrSlot {
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
        tap,
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

/// Resolves the tap's identification at the utterance's ASR final: the
/// speaker's name and description, 未知 when the session runs
/// identification and this utterance matched nothing (or the identification
/// failed or was not ready in time), `None` only when the session runs no
/// identification at all — then the turn's prompt carries no speaker
/// section.
async fn resolve_speaker(pending: Option<PendingSpeaker>, configured: bool) -> Option<Speaker> {
    if !configured {
        return None;
    }
    let identified = match pending {
        Some(result) => tokio::time::timeout(SPEAKER_WAIT, result)
            .await
            .ok()
            .and_then(|resolved| resolved.ok())
            .flatten()
            .filter(|identification| identification.matched),
        None => None,
    };
    Some(identified.map_or_else(
        || Speaker {
            name: UNKNOWN.into(),
            description: String::new(),
        },
        |identification| Speaker {
            name: identification.label,
            description: identification.description,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(value: Option<Identification>) -> PendingSpeaker {
        let (tx, rx) = oneshot::channel();
        tx.send(value).unwrap();
        rx
    }

    fn identification(label: &str, description: &str, matched: bool) -> Option<Identification> {
        Some(Identification {
            label: label.into(),
            description: description.into(),
            similarity: 0.9,
            matched,
        })
    }

    #[tokio::test]
    async fn a_matched_identification_resolves_to_name_and_description() {
        let speaker = resolve_speaker(Some(pending(identification("张三", "同事", true))), true)
            .await
            .unwrap();
        assert_eq!(speaker.name, "张三");
        assert_eq!(speaker.description, "同事");
    }

    #[tokio::test]
    async fn an_unmatched_identification_resolves_to_unknown() {
        let speaker = resolve_speaker(Some(pending(identification("张三", "同事", false))), true)
            .await
            .unwrap();
        assert_eq!(speaker.name, UNKNOWN);
        assert_eq!(speaker.description, "");
    }

    #[tokio::test]
    async fn a_failed_or_unresolved_identification_resolves_to_unknown() {
        // The identification failed: the sender dropped without a value.
        let (tx, rx) = oneshot::channel();
        drop(tx);
        let speaker = resolve_speaker(Some(rx), true).await.unwrap();
        assert_eq!(speaker.name, UNKNOWN);

        // The result never arrived in time: the wait expires.
        let (_tx, rx) = oneshot::channel::<Option<Identification>>();
        let speaker = resolve_speaker(Some(rx), true).await.unwrap();
        assert_eq!(speaker.name, UNKNOWN);
    }

    #[tokio::test]
    async fn a_configured_session_without_a_result_resolves_to_unknown() {
        // The ASR final landed before the segment closed — no pending tap
        // result, but the session runs identification.
        let speaker = resolve_speaker(None, true).await.unwrap();
        assert_eq!(speaker.name, UNKNOWN);
    }

    #[tokio::test]
    async fn without_identification_the_turn_carries_no_speaker() {
        assert_eq!(resolve_speaker(None, false).await, None);
    }
}
