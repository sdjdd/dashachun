use std::collections::VecDeque;
use std::time::Duration;

use axum::extract::ws::Message;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{debug, trace};

use crate::dto::ws::TtsMessage;

use super::send_json;

const PLAYER_CHANNEL_CAPACITY: usize = 64;
const MAX_PREBUFFER_FRAMES: u32 = 8;

enum PlayerCommand {
    Packet(Vec<u8>),
    Subtitle { start_ms: u64, text: String },
    Start,
    Finish(oneshot::Sender<()>),
    Abort,
}

pub(super) struct Player {
    tx: mpsc::Sender<PlayerCommand>,
}

impl Clone for Player {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl Player {
    pub(super) async fn push(&self, packet: Vec<u8>) {
        if self.tx.send(PlayerCommand::Packet(packet)).await.is_err() {
            trace!("player closed, dropping audio packet");
        }
    }

    pub(super) async fn subtitle(&self, start_ms: u64, text: String) {
        if self
            .tx
            .send(PlayerCommand::Subtitle { start_ms, text })
            .await
            .is_err()
        {
            trace!("player closed, dropping subtitle");
        }
    }

    pub(super) async fn start(&self) {
        let _ = self.tx.send(PlayerCommand::Start).await;
    }

    pub(super) async fn abort(&self) {
        let _ = self.tx.send(PlayerCommand::Abort).await;
    }

    pub(super) async fn finish(&self) {
        let (done_tx, done_rx) = oneshot::channel();
        if self.tx.send(PlayerCommand::Finish(done_tx)).await.is_ok() {
            let _ = done_rx.await;
        }
    }
}

pub(super) fn spawn_player(
    tx: mpsc::Sender<Message>,
    frame_duration_ms: u32,
    prebuffer_ms: u32,
    session_id: String,
) -> Player {
    let (player_tx, player_rx) = mpsc::channel::<PlayerCommand>(PLAYER_CHANNEL_CAPACITY);
    let frame_ms = frame_duration_ms.max(1);
    let frame = Duration::from_millis(u64::from(frame_ms));
    let burst = (prebuffer_ms / frame_ms).clamp(1, MAX_PREBUFFER_FRAMES);
    let offset_ms = u64::from(burst) * u64::from(frame_ms);
    tokio::spawn(run_player(
        player_rx, tx, frame, frame_ms, burst, offset_ms, session_id,
    ));
    Player { tx: player_tx }
}

struct PlayerState {
    queue: VecDeque<Vec<u8>>,
    ack: Option<oneshot::Sender<()>>,
    credits: f64,
    capacity: f64,
    frame_ms: u64,
    offset_ms: u64,
    sent_ms: u64,
    pending: VecDeque<(u64, String)>,
}

impl PlayerState {
    fn reset(&mut self) {
        self.sent_ms = 0;
        self.pending.clear();
    }

    fn enqueue_subtitle(&mut self, start_ms: u64, text: String) {
        let at_ms = start_ms.saturating_add(self.offset_ms);
        self.pending.push_back((at_ms, text));
    }
}

async fn flush_due_subtitles(
    state: &mut PlayerState,
    tx: &mpsc::Sender<Message>,
    session_id: &str,
) {
    while let Some((at_ms, _)) = state.pending.front() {
        if *at_ms > state.sent_ms {
            break;
        }
        let Some((_, text)) = state.pending.pop_front() else {
            break;
        };
        debug!(text = %text, sent_ms = state.sent_ms, "subtitle sent");
        send_json(
            tx,
            &TtsMessage::sentence_start(session_id.to_string(), text),
        )
        .await;
    }
}

async fn run_player(
    mut rx: mpsc::Receiver<PlayerCommand>,
    tx: mpsc::Sender<Message>,
    frame: Duration,
    frame_ms: u32,
    burst: u32,
    offset_ms: u64,
    session_id: String,
) {
    let capacity = f64::from(burst);
    let mut state = PlayerState {
        queue: VecDeque::new(),
        ack: None,
        credits: capacity,
        capacity,
        frame_ms: u64::from(frame_ms),
        offset_ms,
        sent_ms: 0,
        pending: VecDeque::new(),
    };
    let mut last: Option<Instant> = None;
    loop {
        let now = Instant::now();
        if let Some(previous) = last {
            state.credits = (state.credits
                + now.duration_since(previous).as_secs_f64() / frame.as_secs_f64())
            .min(state.capacity);
        }
        last = Some(now);

        if !state.queue.is_empty() {
            if state.credits >= 1.0 {
                let Some(packet) = state.queue.pop_front() else {
                    continue;
                };
                if tx.send(Message::Binary(packet.into())).await.is_err() {
                    return;
                }
                state.credits -= 1.0;
                state.sent_ms = state.sent_ms.saturating_add(state.frame_ms);
                last = Some(Instant::now());
                flush_due_subtitles(&mut state, &tx, &session_id).await;
                continue;
            }
            let wait = frame.mul_f64(1.0 - state.credits);
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                command = rx.recv() => {
                    if !handle_player_command(command, &mut state, &tx, &session_id).await {
                        return;
                    }
                }
            }
            continue;
        }

        if state.ack.is_some() {
            drain_pending_subtitles(&mut state, &tx, &session_id).await;
            if let Some(done) = state.ack.take() {
                let _ = done.send(());
            }
        }
        if !handle_player_command(rx.recv().await, &mut state, &tx, &session_id).await {
            return;
        }
    }
}

async fn drain_pending_subtitles(
    state: &mut PlayerState,
    tx: &mpsc::Sender<Message>,
    session_id: &str,
) {
    while let Some((_, text)) = state.pending.pop_front() {
        debug!(text = %text, "subtitle flushed");
        send_json(
            tx,
            &TtsMessage::sentence_start(session_id.to_string(), text),
        )
        .await;
    }
}

async fn handle_player_command(
    command: Option<PlayerCommand>,
    state: &mut PlayerState,
    tx: &mpsc::Sender<Message>,
    session_id: &str,
) -> bool {
    match command {
        Some(PlayerCommand::Packet(packet)) => state.queue.push_back(packet),
        Some(PlayerCommand::Subtitle { start_ms, text }) => {
            state.enqueue_subtitle(start_ms, text);
            flush_due_subtitles(state, tx, session_id).await;
        }
        Some(PlayerCommand::Start) => state.reset(),
        Some(PlayerCommand::Finish(done)) => state.ack = Some(done),
        Some(PlayerCommand::Abort) => {
            state.queue.clear();
            state.credits = state.capacity;
            state.reset();
        }
        None => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence_start(message: &Message) -> Option<String> {
        let Message::Text(text) = message else {
            return None;
        };
        let value: serde_json::Value = serde_json::from_str(text.as_str()).ok()?;
        (value["type"] == "tts" && value["state"] == "sentence_start")
            .then(|| value["text"].as_str().unwrap_or_default().to_string())
    }

    #[tokio::test]
    async fn player_releases_one_packet_per_frame() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        for i in 0..3u8 {
            player.push(vec![i]).await;
        }
        let finished = tokio::spawn(async move {
            player.finish().await;
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        let first = rx.try_recv().expect("first packet should be released");
        assert!(matches!(first, Message::Binary(data) if data.as_ref() == [0u8]));
        assert!(
            rx.try_recv().is_err(),
            "packets released ahead of frame pacing"
        );

        finished.await.unwrap();
        let mut rest = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let Message::Binary(data) = msg {
                rest.push(data[0]);
            }
        }
        assert_eq!(rest, vec![1, 2]);
    }

    #[tokio::test]
    async fn player_primes_device_queue_with_prebuffer() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 100, "test".into());
        for i in 0..4u8 {
            player.push(vec![i]).await;
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut primed = Vec::new();
        while let Ok(Message::Binary(data)) = rx.try_recv() {
            primed.push(data[0]);
        }
        assert_eq!(primed, vec![0, 1], "expected a two-frame prebuffer burst");
        assert!(rx.try_recv().is_err(), "rest must stay paced");

        let finished = tokio::spawn(async move {
            player.finish().await;
        });
        finished.await.unwrap();
        let mut rest = Vec::new();
        while let Ok(Message::Binary(data)) = rx.try_recv() {
            rest.push(data[0]);
        }
        assert_eq!(rest, vec![2, 3]);
    }

    #[tokio::test]
    async fn player_abort_drops_pending_audio() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        for i in 0..3u8 {
            player.push(vec![i]).await;
        }
        player.abort().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let first = rx.try_recv().expect("first packet should be released");
        assert!(matches!(first, Message::Binary(data) if data.as_ref() == [0u8]));

        let finished = tokio::spawn(async move {
            player.finish().await;
        });
        finished.await.unwrap();
        assert!(
            rx.try_recv().is_err(),
            "buffered packets should be dropped after abort"
        );
    }

    #[tokio::test]
    async fn player_emits_subtitle_after_reaching_playback_position() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 100, "test".into());
        player.push(vec![0]).await;
        player.push(vec![1]).await;
        player.push(vec![2]).await;
        player.subtitle(100, "hello".into()).await;

        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut seen = Vec::new();
        while let Ok(message) = rx.try_recv() {
            seen.push(message);
        }
        assert_eq!(seen.len(), 2, "prebuffer burst must precede the subtitle");
        assert!(seen.iter().all(|m| matches!(m, Message::Binary(_))));

        let finished = tokio::spawn(async move {
            player.finish().await;
        });
        finished.await.unwrap();

        let mut subtitles = Vec::new();
        while let Ok(message) = rx.try_recv() {
            if let Some(text) = sentence_start(&message) {
                subtitles.push(text);
            }
        }
        assert_eq!(subtitles, vec!["hello".to_string()]);
    }

    #[tokio::test]
    async fn player_flushes_subtitle_before_finish_ack() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        player.push(vec![0]).await;
        player.subtitle(10_000, "late".into()).await;

        player.finish().await;

        let mut subtitles = Vec::new();
        while let Ok(message) = rx.try_recv() {
            if let Some(text) = sentence_start(&message) {
                subtitles.push(text);
            }
        }
        assert_eq!(subtitles, vec!["late".to_string()]);
    }

    #[tokio::test]
    async fn player_abort_drops_pending_subtitle() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        player.push(vec![0]).await;
        player.subtitle(10_000, "gone".into()).await;
        player.abort().await;

        player.finish().await;

        while let Ok(message) = rx.try_recv() {
            assert!(
                sentence_start(&message).is_none(),
                "subtitle should be dropped"
            );
        }
    }

    #[tokio::test]
    async fn cloned_player_handle_clears_pending_subtitle() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        let session_handle = player.clone();

        player.subtitle(10_000, "gone".into()).await;
        session_handle.abort().await;

        player.finish().await;
        while let Ok(message) = rx.try_recv() {
            assert!(
                sentence_start(&message).is_none(),
                "abort via a cloned handle must drop pending subtitles"
            );
        }
    }
}
