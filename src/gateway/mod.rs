mod player;
mod session;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, trace};

use crate::agent::Agent;
use crate::state::ServerState;

use session::Session;

const WRITER_CHANNEL_CAPACITY: usize = 32;

pub struct Gateway {
    state: ServerState,
    agent: Arc<dyn Agent>,
    shutdown: watch::Receiver<bool>,
}

impl Gateway {
    pub fn new(state: ServerState, agent: Arc<dyn Agent>, shutdown: watch::Receiver<bool>) -> Self {
        Self {
            state,
            agent,
            shutdown,
        }
    }

    pub async fn run(self, socket: WebSocket) {
        let Self {
            state,
            agent,
            shutdown,
        } = self;
        info!("device connected");
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = mpsc::channel::<Message>(WRITER_CHANNEL_CAPACITY);

        let writer = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if SinkExt::send(&mut sink, msg).await.is_err() {
                    break;
                }
            }
        });

        let mut session =
            Session::new(tx.clone(), agent, state.config.server.playback_prebuffer_ms);
        let grace = Duration::from_millis(state.config.server.shutdown_grace_ms);
        let mut shutdown = shutdown;

        let mut shutdown_requested = false;
        loop {
            tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        shutdown_requested = true;
                        break;
                    }
                }
                result = stream.next() => {
                    let Some(result) = result else { break };
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
                            trace!(%txt, "inbound text");
                            session.handle_text(txt.as_str()).await;
                        }
                        Message::Binary(data) => {
                            trace!(len = data.len(), "inbound binary");
                            session.handle_binary(&data).await;
                        }

                        _ => {}
                    }
                }
            }
        }

        session.shutdown(grace).await;
        if shutdown_requested {
            let _ = tx.send(Message::Close(None)).await;
        }
        drop(session);
        drop(tx);
        let _ = writer.await;
        info!("device disconnected");
    }
}

pub(crate) async fn send_json<T: serde::Serialize>(tx: &mpsc::Sender<Message>, value: &T) {
    match serde_json::to_string(value) {
        Ok(payload) => {
            let _ = tx.send(Message::text(payload)).await;
        }
        Err(err) => error!(%err, "failed to serialize message"),
    }
}
