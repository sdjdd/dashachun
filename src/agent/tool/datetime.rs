use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use time::{OffsetDateTime, UtcOffset, Weekday};

use crate::agent::ToolSpec;

use super::{ToolHandler, ToolOutcome, params_schema};

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GetDateTimeArgs {}

pub struct GetDateTime {
    offset: UtcOffset,
}

impl GetDateTime {
    pub fn new(timezone_offset_minutes: i32) -> Self {
        Self {
            offset: offset_from_minutes(timezone_offset_minutes),
        }
    }
}

#[async_trait::async_trait]
impl ToolHandler for GetDateTime {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_datetime".to_string(),
            description: "Get the current local date, time and weekday. Use it when the user asks \
                for the current time or date, or needs to reason about today, now, or relative \
                days."
                .to_string(),
            parameters: params_schema::<GetDateTimeArgs>(),
        }
    }

    async fn call(&self, _args: &Value) -> Result<ToolOutcome, String> {
        let now = OffsetDateTime::now_utc().to_offset(self.offset);
        Ok(ToolOutcome {
            content: describe(now),
            output: None,
            needs_reply: true,
        })
    }
}

fn offset_from_minutes(minutes: i32) -> UtcOffset {
    UtcOffset::from_whole_seconds(minutes.saturating_mul(60)).unwrap_or(UtcOffset::UTC)
}

fn describe(now: OffsetDateTime) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}, {} ({})",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
        weekday_text(now.weekday()),
        offset_text(now.offset()),
    )
}

fn offset_text(offset: UtcOffset) -> String {
    let secs = offset.whole_seconds();
    let abs = secs.abs();
    format!(
        "UTC{}{:02}:{:02}",
        if secs < 0 { '-' } else { '+' },
        abs / 3600,
        (abs % 3600) / 60
    )
}

fn weekday_text(weekday: Weekday) -> &'static str {
    match weekday {
        Weekday::Monday => "Monday",
        Weekday::Tuesday => "Tuesday",
        Weekday::Wednesday => "Wednesday",
        Weekday::Thursday => "Thursday",
        Weekday::Friday => "Friday",
        Weekday::Saturday => "Saturday",
        Weekday::Sunday => "Sunday",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn at(unix_secs: i64, hours: i8, minutes: i8) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(unix_secs)
            .unwrap()
            .to_offset(UtcOffset::from_hms(hours, minutes, 0).unwrap())
    }

    #[test]
    fn formats_local_datetime() {
        let text = describe(at(0, 8, 0));
        assert_eq!(text, "1970-01-01 08:00:00, Thursday (UTC+08:00)");
    }

    #[test]
    fn negative_offset_rolls_date_back() {
        let text = describe(at(0, -5, 0));
        assert_eq!(text, "1969-12-31 19:00:00, Wednesday (UTC-05:00)");
    }

    #[test]
    fn half_hour_offset() {
        let text = describe(at(0, 5, 30));
        assert_eq!(text, "1970-01-01 05:30:00, Thursday (UTC+05:30)");
    }

    #[test]
    fn offset_from_minutes_variants() {
        assert_eq!(offset_from_minutes(480).whole_seconds(), 28_800);
        assert_eq!(offset_from_minutes(-390).whole_seconds(), -23_400);
        assert_eq!(offset_from_minutes(0).whole_seconds(), 0);
        assert_eq!(offset_from_minutes(1_000_000).whole_seconds(), 0);
    }

    #[tokio::test]
    async fn call_ignores_args() {
        let tool = GetDateTime::new(480);
        assert!(tool.call(&json!({})).await.is_ok());
        assert!(tool.call(&json!({ "unexpected": true })).await.is_ok());
    }

    #[test]
    fn spec_schema_matches_type() {
        let spec = GetDateTime::new(480).spec();
        assert_eq!(spec.name, "get_datetime");
        assert_eq!(spec.parameters["type"], "object");
        assert_eq!(spec.parameters["additionalProperties"], false);
    }
}
