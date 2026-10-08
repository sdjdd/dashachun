mod player;
mod session;

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, trace, warn};

use crate::agent::Agent;
use crate::state::ServerState;

use session::Session;

const WRITER_CHANNEL_CAPACITY: usize = 32;
const PING_INTERVAL: Duration = Duration::from_secs(10);
const PEER_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Gateway {
    state: ServerState,
    agent: Arc<dyn Agent>,
    shutdown: watch::Receiver<bool>,
    ping_interval: Duration,
    peer_idle_timeout: Duration,
}

impl Gateway {
    pub fn new(state: ServerState, agent: Arc<dyn Agent>, shutdown: watch::Receiver<bool>) -> Self {
        Self {
            state,
            agent,
            shutdown,
            ping_interval: PING_INTERVAL,
            peer_idle_timeout: PEER_IDLE_TIMEOUT,
        }
    }

    #[cfg(test)]
    fn with_heartbeat_timing(
        state: ServerState,
        agent: Arc<dyn Agent>,
        shutdown: watch::Receiver<bool>,
        ping_interval: Duration,
        peer_idle_timeout: Duration,
    ) -> Self {
        Self {
            state,
            agent,
            shutdown,
            ping_interval,
            peer_idle_timeout,
        }
    }

    pub async fn run(self, socket: WebSocket) {
        let Self {
            state,
            agent,
            shutdown,
            ping_interval,
            peer_idle_timeout,
        } = self;
        info!("device connected");
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = mpsc::channel::<Message>(WRITER_CHANNEL_CAPACITY);

        let writer = tokio::spawn(async move {
            let mut tick = tokio::time::interval_at(
                tokio::time::Instant::now() + ping_interval,
                ping_interval,
            );
            loop {
                tokio::select! {
                    msg = rx.recv() => {
                        let Some(msg) = msg else { break };
                        if SinkExt::send(&mut sink, msg).await.is_err() {
                            break;
                        }
                    }
                    _ = tick.tick() => {
                        if SinkExt::send(&mut sink, Message::Ping(Bytes::new()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });

        let mut session =
            Session::new(tx.clone(), agent, state.config.server.playback_prebuffer_ms);
        let grace = Duration::from_millis(state.config.server.shutdown_grace_ms);
        let mut shutdown = shutdown;

        let mut shutdown_requested = false;
        let mut rejected = false;
        let mut last_seen = tokio::time::Instant::now();
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
                    last_seen = tokio::time::Instant::now();
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
                            if session.handle_text(txt.as_str()).await.is_break() {
                                rejected = true;
                                break;
                            }
                        }
                        Message::Binary(data) => {
                            trace!(len = data.len(), "inbound binary");
                            session.handle_binary(&data).await;
                        }

                        _ => {}
                    }
                }
                _ = tokio::time::sleep_until(last_seen + peer_idle_timeout) => {
                    info!(
                        timeout = ?peer_idle_timeout,
                        "no traffic from device, closing connection"
                    );
                    break;
                }
            }
        }

        session.shutdown(grace).await;
        if shutdown_requested || rejected {
            let _ = tx.send(Message::Close(None)).await;
        }
        drop(session);
        drop(tx);
        let mut writer = writer;
        if tokio::time::timeout(grace, &mut writer).await.is_err() {
            warn!("writer did not stop in time, dropping the socket");
            writer.abort();
        }
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

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::extract::ws::WebSocketUpgrade;
    use axum::{Router, routing::get};
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    use super::Gateway;
    use crate::agent::{
        Agent, AgentFactory, AgentInputStream, AgentOutputStream, AgentSession, AgentStore,
        InMemMemoryFactory, ToolRegistry,
    };
    use crate::asr::StubAsr;
    use crate::config::{AppConfig, DeviceConfig, OtaConfig, ServerConfig};
    use crate::device::DeviceStore;
    use crate::llm::StubLlm;
    use crate::state::ServerState;
    use crate::tts::StubTts;
    use crate::vad::{SileroVadFactory, VadConfig};

    const HELLO: &str = r#"{"type":"hello","version":1,"transport":"websocket","features":{"mcp":false,"aec":false}}"#;

    struct NullAgent;

    impl Agent for NullAgent {
        fn run(&self, _session: AgentSession, _input: AgentInputStream) -> AgentOutputStream {
            Box::pin(futures_util::stream::empty())
        }
    }

    fn test_state() -> (ServerState, Arc<dyn Agent>) {
        // The gateway never touches the database; a lazy pool keeps the tests
        // free of Postgres.
        let pool =
            sqlx::PgPool::connect_lazy("postgres://unused@localhost/unused").expect("lazy pool");
        let state = ServerState::new(
            AppConfig {
                server: ServerConfig {
                    bind_addr: "127.0.0.1:0".into(),
                    playback_prebuffer_ms: 180,
                    shutdown_grace_ms: 5000,
                },
                ota: OtaConfig {
                    websocket_url: None,
                    timezone_offset: 480,
                },
                device: DeviceConfig {
                    activation_ttl_secs: 600,
                },
            },
            Arc::new(AgentFactory::new(
                Arc::new(StubAsr::default()),
                Arc::new(StubLlm::default()),
                Arc::new(StubTts),
                Arc::new(SileroVadFactory::new(VadConfig::default())),
                Arc::new(InMemMemoryFactory),
                Arc::new(ToolRegistry::new(Vec::new())),
                AgentStore::new(pool.clone()),
                pool.clone(),
            )),
            DeviceStore::new(pool),
        );
        (state, Arc::new(NullAgent))
    }

    async fn spawn_gateway(
        state: ServerState,
        agent: Arc<dyn Agent>,
        ping_interval: Duration,
        peer_idle_timeout: Duration,
    ) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/gateway",
            get(move |ws: WebSocketUpgrade| {
                let state = state.clone();
                let agent = agent.clone();
                async move {
                    let shutdown = state.shutdown_signal();
                    ws.on_upgrade(move |socket| {
                        Gateway::with_heartbeat_timing(
                            state,
                            agent,
                            shutdown,
                            ping_interval,
                            peer_idle_timeout,
                        )
                        .run(socket)
                    })
                }
            }),
        );
        tokio::spawn(axum::serve(listener, app).into_future());
        addr
    }

    async fn connect(
        addr: SocketAddr,
    ) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    {
        tokio_tungstenite::connect_async(format!("ws://{addr}/gateway"))
            .await
            .unwrap()
            .0
    }

    async fn hello_ack(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) {
        ws.send(Message::text(HELLO)).await.unwrap();
        loop {
            let msg = ws.next().await.unwrap().unwrap();
            if matches!(msg, Message::Text(_)) {
                return;
            }
        }
    }

    #[tokio::test]
    async fn ponging_device_survives_the_idle_deadline() {
        let (state, agent) = test_state();
        let addr = spawn_gateway(
            state,
            agent,
            Duration::from_millis(50),
            Duration::from_millis(300),
        )
        .await;

        let mut ws = connect(addr).await;
        hello_ack(&mut ws).await;

        // Reading drives tungstenite's automatic pong replies, which must
        // keep the connection alive well past the idle deadline.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
        let mut pings = 0;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, ws.next()).await {
                Ok(Some(Ok(Message::Ping(_)))) => pings += 1,
                Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
                    panic!("connection closed despite answering pings");
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(err))) => panic!("websocket error: {err}"),
                Err(_) => break,
            }
        }
        assert!(pings >= 2, "expected periodic pings, got {pings}");
    }

    #[tokio::test]
    async fn silent_device_is_reaped() {
        let (state, agent) = test_state();
        let addr = spawn_gateway(
            state,
            agent,
            Duration::from_millis(50),
            Duration::from_millis(300),
        )
        .await;

        let mut ws = connect(addr).await;
        hello_ack(&mut ws).await;

        // Stop reading: the auto-pong machinery never runs, so the server
        // sees no inbound traffic at all.
        tokio::time::sleep(Duration::from_millis(500)).await;

        let closed = loop {
            match ws.next().await {
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break true,
                Some(Ok(_)) => {}
            }
        };
        assert!(closed, "server did not reap the silent connection");
    }
}
