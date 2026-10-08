use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, Stream, StreamExt};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::handshake::client::Request;
use tracing::debug;

use super::CANCEL_TIMEOUT;
use super::Connection;
use super::next_matching;
use super::protocol;
use crate::agent::TtsError;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const CONN_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const IDLE_SWEEP_INTERVAL: Duration = Duration::from_secs(5);
const MAX_IDLE_CONNS: usize = 16;

pub(crate) struct Parked {
    pub(crate) sink: SplitSink<Connection, Message>,
    pub(crate) stream: SplitStream<Connection>,
    pub(crate) connect_id: String,
}

pub(crate) struct ConnSpec {
    pub(crate) endpoint: String,
    pub(crate) api_key: String,
    pub(crate) resource_id: String,
}

impl ConnSpec {
    fn build_request(&self) -> Result<(Request, String), TtsError> {
        let mut request = self
            .endpoint
            .as_str()
            .into_client_request()
            .map_err(|err| TtsError::from(format!("invalid endpoint: {err}")))?;
        let connect_id = uuid::Uuid::new_v4().to_string();
        let headers = request.headers_mut();
        for (name, value) in [
            ("X-Api-Key", self.api_key.as_str()),
            ("X-Api-Resource-Id", self.resource_id.as_str()),
            ("X-Api-Connect-Id", connect_id.as_str()),
            ("X-Control-Require-Usage-Tokens-Return", "*"),
        ] {
            let value = value
                .parse()
                .map_err(|err| TtsError::from(format!("invalid header {name}: {err}")))?;
            headers.insert(name, value);
        }
        Ok((request, connect_id))
    }
}

pub(crate) struct PoolTiming {
    pub(crate) idle_timeout: Duration,
    pub(crate) sweep_interval: Duration,
    pub(crate) max_idle: usize,
}

impl Default for PoolTiming {
    fn default() -> Self {
        Self {
            idle_timeout: CONN_IDLE_TIMEOUT,
            sweep_interval: IDLE_SWEEP_INTERVAL,
            max_idle: MAX_IDLE_CONNS,
        }
    }
}

struct Entry {
    conn: Parked,
    parked_at: Instant,
}

pub(crate) struct ConnPool {
    spec: ConnSpec,
    idle: Mutex<Vec<Entry>>,
    max_idle: usize,
}

impl ConnPool {
    pub(crate) fn new(spec: ConnSpec, timing: PoolTiming) -> Arc<Self> {
        let pool = Arc::new(Self {
            spec,
            idle: Mutex::new(Vec::new()),
            max_idle: timing.max_idle,
        });
        spawn_idle_sweeper(pool.clone(), timing.idle_timeout, timing.sweep_interval);
        pool
    }

    pub(crate) fn checkin(&self, conn: Parked) {
        let overflow = {
            let mut idle = self.idle.lock().unwrap();
            if idle.len() >= self.max_idle {
                Some(conn)
            } else {
                idle.push(Entry {
                    conn,
                    parked_at: Instant::now(),
                });
                None
            }
        };
        if let Some(conn) = overflow {
            debug!(connect_id = %conn.connect_id, "tts pool full, closing connection");
            tokio::spawn(close_conn(conn));
        }
    }

    pub(crate) async fn acquire(&self) -> Result<Parked, TtsError> {
        loop {
            let Some(mut parked) = self.checkout() else {
                return self.open().await;
            };
            if connection_alive(&mut parked) {
                debug!(connect_id = %parked.connect_id, "tts reusing pooled connection");
                return Ok(parked);
            }
            debug!(connect_id = %parked.connect_id, "tts pooled connection stale, dropping");
        }
    }

    fn checkout(&self) -> Option<Parked> {
        self.idle.lock().unwrap().pop().map(|entry| entry.conn)
    }

    async fn open(&self) -> Result<Parked, TtsError> {
        let (request, connect_id) = self.spec.build_request()?;
        let socket = connect(request).await?;
        let (sink, stream) = socket.split();
        let mut parked = Parked {
            sink,
            stream,
            connect_id,
        };
        parked
            .sink
            .send(Message::Binary(protocol::start_connection().into()))
            .await
            .map_err(|err| TtsError::from(format!("send start connection: {err}")))?;
        next_matching(
            &mut parked.stream,
            protocol::MSG_FULL_SERVER_RESPONSE,
            protocol::EVENT_CONNECTION_STARTED,
        )
        .await?;
        debug!(connect_id = %parked.connect_id, "tts connection started");
        Ok(parked)
    }

    fn expire(&self, idle_timeout: Duration) -> Vec<Parked> {
        let mut idle = self.idle.lock().unwrap();
        let (keep, expired) = idle
            .drain(..)
            .partition(|entry| entry.parked_at.elapsed() < idle_timeout);
        *idle = keep;
        expired.into_iter().map(|entry| entry.conn).collect()
    }

    #[cfg(test)]
    pub(crate) fn parked_count(&self) -> usize {
        self.idle.lock().unwrap().len()
    }
}

fn spawn_idle_sweeper(pool: Arc<ConnPool>, idle_timeout: Duration, sweep_interval: Duration) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(sweep_interval);
        loop {
            tick.tick().await;
            for conn in pool.expire(idle_timeout) {
                debug!(
                    connect_id = %conn.connect_id,
                    "tts idle connection expired, closing"
                );
                close_conn(conn).await;
            }
        }
    });
}

async fn close_conn(conn: Parked) {
    let Parked {
        mut sink,
        mut stream,
        ..
    } = conn;
    if sink
        .send(Message::Binary(protocol::finish_connection().into()))
        .await
        .is_err()
    {
        return;
    }
    let _ = tokio::time::timeout(
        CANCEL_TIMEOUT,
        next_matching(
            &mut stream,
            protocol::MSG_FULL_SERVER_RESPONSE,
            protocol::EVENT_CONNECTION_FINISHED,
        ),
    )
    .await;
}

async fn connect(request: Request) -> Result<Connection, TtsError> {
    let (socket, _response) = tokio::time::timeout(CONNECT_TIMEOUT, async {
        tokio_tungstenite::connect_async(request).await
    })
    .await
    .map_err(|_| TtsError::from("connect timeout"))?
    .map_err(|err| TtsError::from(format!("connect failed: {err}")))?;

    debug!("tts websocket connected");
    Ok(socket)
}

/// Checks a parked connection for a server-side close while it sat idle: any
/// frames already delivered (stale session leftovers, keepalives) are drained
/// and discarded. `false` means the connection died and must not be reused.
fn connection_alive(conn: &mut Parked) -> bool {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match Pin::new(&mut conn.stream).poll_next(&mut cx) {
            Poll::Pending => return true,
            Poll::Ready(None)
            | Poll::Ready(Some(Err(_)))
            | Poll::Ready(Some(Ok(Message::Close(_)))) => return false,
            Poll::Ready(Some(Ok(_))) => continue,
        }
    }
}
