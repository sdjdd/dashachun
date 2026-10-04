use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use crate::dto::ws::{
    Abort, AudioParams, InboundMessage, Listen, Mcp, ServerHello, SttMessage, TtsMessage,
};
use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new().route("/gateway", get(handle_device_connect))
}

async fn handle_device_connect(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_device_socket(socket, state))
}

async fn handle_device_socket(socket: WebSocket, _state: AppState) {
    info!("device connected");
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<Message>(32);

    let writer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if SinkExt::send(&mut sink, msg).await.is_err() {
                break;
            }
        }
    });

    let mut session_id: Option<String> = None;
    while let Some(result) = stream.next().await {
        let msg = match result {
            Ok(Message::Close(_)) => break,
            Ok(msg) => msg,
            Err(err) => {
                debug!(%err, "websocket stream ended");
                break;
            }
        };

        match msg {
            Message::Text(txt) => {
                if let Err(err) = handle_text(&txt, &mut session_id, &tx).await {
                    warn!(%err, "invalid message");
                }
            }
            Message::Binary(data) => {
                trace!(len = data.len(), "received binary audio frame");
            }
            _ => {}
        }
    }

    drop(tx);
    let _ = writer.await;
    info!("device disconnected");
}

async fn handle_text(
    txt: &Utf8Bytes,
    session_id: &mut Option<String>,
    tx: &mpsc::Sender<Message>,
) -> Result<(), serde_json::Error> {
    let message: InboundMessage = serde_json::from_str(txt.as_str())?;

    match message {
        InboundMessage::Hello(hello) => {
            let id = Uuid::new_v4().to_string();
            info!(
                session_id = %id,
                version = hello.version,
                transport = %hello.transport,
                mcp = hello.features.mcp,
                aec = hello.features.aec,
                "device hello"
            );
            send_json(tx, &ServerHello::new(id.clone(), AudioParams::default())).await;
            *session_id = Some(id);
        }
        InboundMessage::Listen(listen) => handle_listen(listen, session_id, tx).await,
        InboundMessage::Abort(abort) => handle_abort(abort, session_id, tx).await,
        InboundMessage::Mcp(mcp) => handle_mcp(mcp, session_id),
    }

    Ok(())
}

async fn handle_listen(listen: Listen, session_id: &Option<String>, tx: &mpsc::Sender<Message>) {
    let Some(session_id) = session_id else {
        warn!("listen message before hello, ignoring");
        return;
    };

    match listen.state.as_str() {
        "start" => {
            info!(session_id, mode = ?listen.mode, "device started listening");
        }
        "detect" => {
            info!(session_id, text = ?listen.text, "wake word detected");
        }
        "stop" => {
            info!(session_id, "device stopped listening");
            run_stub_pipeline(session_id, tx).await;
        }
        other => warn!(%other, "unknown listen state"),
    }
}

async fn handle_abort(abort: Abort, session_id: &Option<String>, _tx: &mpsc::Sender<Message>) {
    info!(session_id = ?session_id, reason = ?abort.reason, "abort requested");
}

fn handle_mcp(mcp: Mcp, session_id: &Option<String>) {
    debug!(session_id = ?session_id, payload = %mcp.payload, "mcp message");
}

async fn run_stub_pipeline(session_id: &str, tx: &mpsc::Sender<Message>) {
    info!(session_id, "stub pipeline: stt -> tts start/stop");
    send_json(
        tx,
        &SttMessage::new(session_id.to_string(), "你好".to_string()),
    )
    .await;
    send_json(tx, &TtsMessage::start(session_id.to_string())).await;
    send_json(
        tx,
        &TtsMessage::sentence_start(session_id.to_string(), "你好".to_string()),
    )
    .await;
    send_json(tx, &TtsMessage::stop(session_id.to_string())).await;
}

async fn send_json<T: serde::Serialize>(tx: &mpsc::Sender<Message>, value: &T) {
    match serde_json::to_string(value) {
        Ok(payload) => {
            let _ = tx.send(Message::text(payload)).await;
        }
        Err(err) => error!(%err, "failed to serialize message"),
    }
}
