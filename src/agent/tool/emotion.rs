use std::fmt;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::agent::AgentOutput;
use crate::llm::ToolSpec;

use super::{ToolHandler, ToolOutcome, params_schema};

macro_rules! emotions {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
        pub enum Emotion {
            $(#[serde(rename = $name)] $variant),+
        }

        impl Emotion {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Emotion::$variant => $name),+
                }
            }
        }
    };
}

emotions! {
    Happy => "happy",
    Laughing => "laughing",
    Funny => "funny",
    Loving => "loving",
    Embarrassed => "embarrassed",
    Confident => "confident",
    Delicious => "delicious",
    Sad => "sad",
    Crying => "crying",
    Sleepy => "sleepy",
    Silly => "silly",
    Angry => "angry",
    Surprised => "surprised",
    Shocked => "shocked",
    Thinking => "thinking",
    Winking => "winking",
    Relaxed => "relaxed",
    Confused => "confused",
    Neutral => "neutral",
    Idle => "idle",
    Robot2 => "robot_2",
    LowBattery => "low_battery",
    BatteryConnected => "battery_connected",
}

impl fmt::Display for Emotion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetEmotionArgs {
    /// The expression to display.
    emotion: Emotion,
}

#[derive(Default)]
pub struct SetEmotion;

#[async_trait::async_trait]
impl ToolHandler for SetEmotion {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "set_emotion".to_string(),
            description: "Set the assistant's facial expression before replying. Call this \
                first, then produce the spoken reply. Choose the emotion that best matches the \
                reply's tone."
                .to_string(),
            parameters: params_schema::<SetEmotionArgs>(),
        }
    }

    async fn call(&self, args: &Value) -> Result<ToolOutcome, String> {
        let args: SetEmotionArgs =
            serde_json::from_value(args.clone()).map_err(|err| err.to_string())?;
        let emotion = args.emotion;
        Ok(ToolOutcome {
            content: format!("emotion set to {emotion}"),
            output: Some(AgentOutput::Emotion {
                emotion: emotion.to_string(),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sets_known_emotion() {
        let outcome = SetEmotion
            .call(&serde_json::json!({ "emotion": "happy" }))
            .await
            .unwrap();
        assert_eq!(outcome.content, "emotion set to happy");
        assert!(matches!(
            outcome.output,
            Some(AgentOutput::Emotion { emotion }) if emotion == "happy"
        ));
    }

    #[tokio::test]
    async fn rejects_unknown_emotion() {
        assert!(
            SetEmotion
                .call(&serde_json::json!({ "emotion": "bogus" }))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn rejects_missing_field() {
        assert!(SetEmotion.call(&serde_json::json!({})).await.is_err());
    }

    #[tokio::test]
    async fn rejects_unknown_field() {
        assert!(
            SetEmotion
                .call(&serde_json::json!({ "emotion": "happy", "extra": 1 }))
                .await
                .is_err()
        );
    }

    #[test]
    fn spec_schema_matches_type() {
        let spec = SetEmotion.spec();
        assert_eq!(spec.name, "set_emotion");
        assert_eq!(spec.parameters["type"], "object");
        let emotions = spec.parameters["properties"]["emotion"]["enum"]
            .as_array()
            .unwrap();
        assert!(emotions.contains(&serde_json::json!("happy")));
        assert!(emotions.contains(&serde_json::json!("battery_connected")));
        assert_eq!(spec.parameters["required"], serde_json::json!(["emotion"]));
        assert_eq!(spec.parameters["additionalProperties"], false);
    }

    #[test]
    fn all_emotions_round_trip() {
        for emotion in [
            Emotion::Happy,
            Emotion::Robot2,
            Emotion::LowBattery,
            Emotion::BatteryConnected,
        ] {
            let args = serde_json::json!({ "emotion": emotion.as_str() });
            let parsed: SetEmotionArgs = serde_json::from_value(args).unwrap();
            assert_eq!(parsed.emotion, emotion);
        }
    }
}
