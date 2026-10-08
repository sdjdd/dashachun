use std::sync::Mutex;
use std::time::{Duration, Instant};

use futures_util::stream::{SplitSink, SplitStream};
use tokio_tungstenite::tungstenite::Message;

use super::Connection;

pub(crate) struct Parked {
    pub(crate) sink: SplitSink<Connection, Message>,
    pub(crate) stream: SplitStream<Connection>,
    pub(crate) connect_id: String,
}

struct Entry {
    conn: Parked,
    parked_at: Instant,
}

pub(crate) struct ConnPool {
    idle: Mutex<Vec<Entry>>,
    max_idle: usize,
}

impl ConnPool {
    pub(crate) fn new(max_idle: usize) -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
            max_idle,
        }
    }

    pub(crate) fn checkin(&self, conn: Parked) -> Option<Parked> {
        let mut idle = self.idle.lock().unwrap();
        if idle.len() >= self.max_idle {
            return Some(conn);
        }
        idle.push(Entry {
            conn,
            parked_at: Instant::now(),
        });
        None
    }

    pub(crate) fn checkout(&self) -> Option<Parked> {
        self.idle.lock().unwrap().pop().map(|entry| entry.conn)
    }

    pub(crate) fn expire(&self, idle_timeout: Duration) -> Vec<Parked> {
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
