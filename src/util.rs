//! Tiny shared plumbing with no home of its own.

use std::pin::Pin;

use futures_util::Stream;
use tokio::sync::mpsc;

/// Streams a channel's messages, ending when the channel closes. The one
/// pump behind every `mpsc`-backed stream boundary — provider events,
/// agent stages, the gateway's agent input.
pub fn recv_stream<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }))
}

/// The unbounded-channel twin of [`recv_stream`].
pub fn recv_unbounded_stream<T: Send + 'static>(
    rx: mpsc::UnboundedReceiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }))
}
