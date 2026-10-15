//! The listening leg: the device input, the VAD and the per-utterance ASR
//! tasks for the whole session. Forwards `Stt`/`Mcp`/`Error` directly and
//! hands every final utterance to the reply stage together with a fresh
//! turn token and the utterance's voiceprint attribution — resolved by the
//! ASR task itself, so the row it stores carries the same speaker the turn
//! is labeled with — cancelling the previous one first.

use std::sync::Arc;
use std::time::Duration;

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
use crate::voiceprint::{SAMPLE_RATE, SpeakerMatcher, SpeakerResult, SpeakerTap};

/// How long the ASR final waits for the tap's identification before the
/// utterance stores unlabeled and the turn starts unlabeled. The tap
/// finishes at `SpeechEnd` and the embed takes tens of milliseconds, so
/// the result is usually already resolved when the final arrives — the
/// wait only bounds a rare tail (a long segment, the model mutex held by
/// an enrollment); an expiry stores and starts the turn without
/// attribution, which still logs.
const SPEAKER_WAIT: Duration = Duration::from_millis(250);

/// What one per-utterance ASR task reports back to the listening stage,
/// tagged with its epoch so superseded results are dropped. The final
/// carries the speaker the task resolved (and stored) for its utterance.
enum AsrUpdate {
    Partial {
        epoch: u64,
        text: String,
    },
    Final {
        epoch: u64,
        text: String,
        speaker: Option<i64>,
    },
    Error {
        epoch: u64,
        message: String,
    },
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
    // The identification pipeline is a per-session capability: the matcher
    // exists only when the agent has enrolled speakers and the model
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
                    AsrUpdate::Final {
                        text, speaker, ..
                    } => {
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
) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id = %deps.session.id, at_ms, "speech start");
                *epoch += 1;
                let capture = deps
                    .capture
                    .map(|capture| capture.start_utterance(deps.session));
                // The tap's capture side rides the slot (pushed samples,
                // finished at SpeechEnd); its result channel goes straight
                // to the ASR task, which resolves it at the final — so a
                // final whose turn was superseded still stores with the
                // speaker it identified.
                let (tap, speaker_rx) = match deps.speaker {
                    Some(speaker) => {
                        let (tap, rx) = speaker.tap(deps.session);
                        (Some(tap), Some(rx))
                    }
                    None => (None, None),
                };
                *slot = Some(spawn_asr(&deps, *epoch, capture, tap, speaker_rx));
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
                    current.detach();
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
    /// complete, identification does not wait for the ASR final; the tap's
    /// result channel is already in the task's hands.
    fn detach(mut self) {
        self.task.detach();
        if let Some(tap) = self.tap.take() {
            tap.finish();
        }
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
    speaker_rx: Option<SpeakerResult>,
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
        speaker_rx,
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
    speaker_rx: Option<SpeakerResult>,
) {
    let mut events = asr.transcribe(audio, cancel.clone());
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(AsrEvent::Partial { text }) => AsrUpdate::Partial { epoch, text },
            Ok(AsrEvent::Final { text }) => {
                // The utterance is real speech even when the turn was
                // superseded, so the row is stored regardless of the cancel
                // token; the capture learns its message id through the
                // memory's hook, not from here. The speaker resolves before
                // the store: the row carries the same attribution the turn
                // is labeled with.
                let mut speaker = None;
                if !cancel.is_cancelled() && !text.is_empty() {
                    speaker = resolve_speaker(speaker_rx).await;
                    match memory.store_utterance(&session, &text, speaker).await {
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
                let _ = asr_tx
                    .send(AsrUpdate::Final {
                        epoch,
                        text,
                        speaker,
                    })
                    .await;
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
/// matched speaker's speakers row id, the one the row is stored with and
/// the reply stage prefixes the context's user message with. `None` when
/// this utterance matched nothing (or the identification failed or was
/// not ready in time) — the context marks the message `unknown` — and
/// when the session runs no identification at all, where the turn's prompt
/// carries no roster either.
async fn resolve_speaker(pending: Option<SpeakerResult>) -> Option<i64> {
    let identified = match pending {
        Some(result) => tokio::time::timeout(SPEAKER_WAIT, result)
            .await
            .ok()
            .and_then(|resolved| resolved.ok())
            .flatten()
            .filter(|identification| identification.matched),
        None => None,
    };
    identified.map(|identification| identification.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voiceprint::Identification;
    use tokio::sync::oneshot;

    fn pending(value: Option<Identification>) -> SpeakerResult {
        let (tx, rx) = oneshot::channel();
        tx.send(value).unwrap();
        rx
    }

    fn identification(id: i64, name: &str, matched: bool) -> Option<Identification> {
        Some(Identification {
            id,
            name: name.into(),
            description: "同事".into(),
            similarity: 0.9,
            matched,
        })
    }

    #[tokio::test]
    async fn a_matched_identification_resolves_to_the_speaker_id() {
        assert_eq!(
            resolve_speaker(Some(pending(identification(7, "张三", true)))).await,
            Some(7)
        );
    }

    #[tokio::test]
    async fn below_threshold_or_failed_identifications_stay_unmarked() {
        assert_eq!(
            resolve_speaker(Some(pending(identification(7, "张三", false)))).await,
            None
        );
        // The identification failed: the sender dropped without a value.
        let (tx, rx) = oneshot::channel();
        drop(tx);
        assert_eq!(resolve_speaker(Some(rx)).await, None);

        // The result never arrived in time: the wait expires.
        let (_tx, rx) = oneshot::channel::<Option<Identification>>();
        assert_eq!(resolve_speaker(Some(rx)).await, None);
    }

    #[tokio::test]
    async fn a_session_without_identification_stays_unmarked() {
        // No tap was ever opened — the session runs no identification.
        assert_eq!(resolve_speaker(None).await, None);
    }
}
