use std::convert::Infallible;

use time::{OffsetDateTime, UtcOffset, Weekday};

use super::{NoArgs, Tool};

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
impl Tool for GetDateTime {
    const NAME: &'static str = "get_datetime";
    type Args = NoArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "Get the current local date, time and weekday. Use it when the user asks \
         for the current time or date, or needs to reason about today, now, or relative \
         days."
            .to_string()
    }

    async fn call(&self, _args: NoArgs) -> Result<String, Infallible> {
        let now = OffsetDateTime::now_utc().to_offset(self.offset);
        Ok(describe(now))
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
    use crate::agent::ToolCall;
    use crate::agent::tool::ToolSet;

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
    async fn executes_with_empty_or_object_arguments() {
        let tools = ToolSet::new().tool(GetDateTime::new(480));
        for arguments in ["", "{}"] {
            let outcome = tools
                .execute(&ToolCall {
                    id: "1".into(),
                    name: "get_datetime".into(),
                    arguments: arguments.into(),
                })
                .await;
            assert!(outcome.content.contains("UTC+08:00"));
            assert!(outcome.ok);
        }
    }

    #[test]
    fn spec_schema_matches_type() {
        let spec = ToolSet::new()
            .tool(GetDateTime::new(480))
            .spec("get_datetime")
            .unwrap()
            .clone();
        assert_eq!(spec.parameters["type"], "object");
        assert_eq!(spec.parameters["additionalProperties"], false);
    }
}
