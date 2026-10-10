use std::sync::Arc;
use std::time::Duration;

use redis::aio::MultiplexedConnection;
use redis::{Client, RedisError, Script};
use tokio::sync::OnceCell;

/// Atomic fixed-window counter: `INCR` the key, set the TTL on the first
/// hit of the window, and report both the count and the remaining TTL so
/// the caller can compute `Retry-After` without a second round trip.
const FIXED_WINDOW_LUA: &str = "
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('PTTL', KEYS[1])}
";

pub enum Decision {
    Allowed,
    Limited { retry_after: Duration },
}

/// Redis-backed fixed-window rate limiter. The connection is established
/// lazily on the first `hit` (the `Client` is built eagerly, so a malformed
/// `REDIS_URL` fails fast at boot); runtime backend errors fail open — a
/// limiter outage must not lock users out of the auth endpoints.
#[derive(Clone)]
pub struct RedisLimiter {
    client: Client,
    conn: Arc<OnceCell<MultiplexedConnection>>,
    script: Arc<Script>,
    prefix: Arc<str>,
}

impl RedisLimiter {
    pub fn new(url: &str, prefix: &str) -> Result<Self, RedisError> {
        Ok(Self {
            client: Client::open(url)?,
            conn: Arc::new(OnceCell::new()),
            script: Arc::new(Script::new(FIXED_WINDOW_LUA)),
            prefix: Arc::from(prefix),
        })
    }

    /// Boot-time health check so a mandatory, unreachable Redis exits the
    /// process instead of silently degrading to fail-open.
    pub async fn ping(&self) -> Result<(), RedisError> {
        let mut conn = self.connection().await?;
        redis::cmd("PING").query_async(&mut conn).await
    }

    pub async fn hit(&self, bucket: &str, subject: &str, limit: u32, window: Duration) -> Decision {
        let key = format!("{}:{bucket}:{subject}", self.prefix);
        let result = match self.connection().await {
            Ok(mut conn) => {
                self.script
                    .key(&key)
                    .arg(window.as_millis() as i64)
                    .invoke_async::<(i64, i64)>(&mut conn)
                    .await
            }
            Err(err) => Err(err),
        };
        decide(result, limit, window)
    }

    async fn connection(&self) -> Result<MultiplexedConnection, RedisError> {
        Ok(self
            .conn
            .get_or_try_init(|| self.client.get_multiplexed_async_connection())
            .await?
            .clone())
    }
}

fn decide(result: Result<(i64, i64), RedisError>, limit: u32, window: Duration) -> Decision {
    let (count, ttl_ms) = match result {
        Ok(pair) => pair,
        Err(err) => {
            tracing::warn!(%err, "rate limiter backend error, allowing the request");
            return Decision::Allowed;
        }
    };
    if count <= i64::from(limit) {
        return Decision::Allowed;
    }
    let retry_after = if ttl_ms > 0 {
        Duration::from_millis(ttl_ms as u64)
    } else {
        window
    };
    Decision::Limited { retry_after }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> RedisError {
        std::io::Error::other("boom").into()
    }

    #[test]
    fn under_the_limit_is_allowed() {
        assert!(matches!(
            decide(Ok((1, 60_000)), 10, Duration::from_secs(60)),
            Decision::Allowed
        ));
        assert!(matches!(
            decide(Ok((10, 1)), 10, Duration::from_secs(60)),
            Decision::Allowed
        ));
    }

    #[test]
    fn over_the_limit_reports_the_remaining_window() {
        let Decision::Limited { retry_after } =
            decide(Ok((11, 42_000)), 10, Duration::from_secs(60))
        else {
            panic!("expected a limited decision");
        };
        assert_eq!(retry_after, Duration::from_millis(42_000));
    }

    #[test]
    fn a_nonpositive_ttl_falls_back_to_the_window() {
        let Decision::Limited { retry_after } = decide(Ok((11, -2)), 10, Duration::from_secs(60))
        else {
            panic!("expected a limited decision");
        };
        assert_eq!(retry_after, Duration::from_secs(60));
    }

    #[test]
    fn a_backend_error_fails_open() {
        assert!(matches!(
            decide(Err(err()), 10, Duration::from_secs(60)),
            Decision::Allowed
        ));
    }
}
