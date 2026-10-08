use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::agent::ToolSpec;

use super::{ToolHandler, ToolOutcome, params_schema};

const GEOCODE_URL: &str = "https://geocoding-api.open-meteo.com/v1/search";
const FORECAST_URL: &str = "https://api.open-meteo.com/v1/forecast";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(8);

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GetWeatherArgs {
    /// City name, e.g. Beijing.
    city: String,
}

pub struct GetWeather {
    client: reqwest::Client,
}

impl GetWeather {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self { client }
    }

    async fn geocode(&self, city: &str) -> Result<Location, String> {
        let value: Value = self
            .client
            .get(GEOCODE_URL)
            .query(&[("name", city), ("count", "1"), ("language", "zh")])
            .send()
            .await
            .map_err(|err| format!("geocoding request failed: {err}"))?
            .json()
            .await
            .map_err(|err| format!("geocoding decode failed: {err}"))?;
        parse_location(&value).ok_or_else(|| format!("could not find a city named `{city}`"))
    }

    async fn forecast(&self, location: &Location) -> Result<String, String> {
        let value: Value = self
            .client
            .get(FORECAST_URL)
            .query(&[
                ("latitude", location.latitude.to_string()),
                ("longitude", location.longitude.to_string()),
                ("current", "temperature_2m,weather_code".to_string()),
            ])
            .send()
            .await
            .map_err(|err| format!("forecast request failed: {err}"))?
            .json()
            .await
            .map_err(|err| format!("forecast decode failed: {err}"))?;
        parse_forecast(location, &value)
            .ok_or_else(|| "forecast response was missing current weather".to_string())
    }
}

impl Default for GetWeather {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolHandler for GetWeather {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_weather".to_string(),
            description: "Look up the current weather for a city. Use it when the user asks \
                about the weather."
                .to_string(),
            parameters: params_schema::<GetWeatherArgs>(),
        }
    }

    async fn call(&self, args: &Value) -> Result<ToolOutcome, String> {
        let args: GetWeatherArgs =
            serde_json::from_value(args.clone()).map_err(|err| err.to_string())?;
        let city = args.city.trim();
        if city.is_empty() {
            return Err("city must not be empty".to_string());
        }
        let location = self.geocode(city).await?;
        let content = self.forecast(&location).await?;
        Ok(ToolOutcome {
            content,
            output: None,
            needs_reply: true,
        })
    }
}

struct Location {
    latitude: f64,
    longitude: f64,
    label: String,
}

fn parse_location(value: &Value) -> Option<Location> {
    let first = value.get("results")?.as_array()?.first()?;
    let latitude = first.get("latitude")?.as_f64()?;
    let longitude = first.get("longitude")?.as_f64()?;
    let label = first
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    Some(Location {
        latitude,
        longitude,
        label,
    })
}

fn parse_forecast(location: &Location, value: &Value) -> Option<String> {
    let current = value.get("current")?;
    let temperature = current.get("temperature_2m")?.as_f64()?;
    let code = current.get("weather_code")?.as_i64()?;
    Some(format!(
        "{}：{}，气温 {:.1}°C。",
        location.label,
        weather_text(code),
        temperature
    ))
}

fn weather_text(code: i64) -> &'static str {
    match code {
        0 => "晴",
        1 => "晴间多云",
        2 => "多云",
        3 => "阴",
        45 | 48 => "雾",
        51 | 53 | 55 => "毛毛雨",
        56 | 57 => "冻毛毛雨",
        61 | 63 | 65 => "雨",
        66 | 67 => "冻雨",
        71 | 73 | 75 | 77 => "雪",
        80..=82 => "阵雨",
        85 | 86 => "阵雪",
        95 => "雷阵雨",
        96 | 99 => "雷阵雨伴冰雹",
        _ => "未知天气",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_geocoding_result() {
        let value = json!({
            "results": [
                { "name": "Beijing", "latitude": 39.9075, "longitude": 116.39723 }
            ]
        });
        let location = parse_location(&value).unwrap();
        assert_eq!(location.label, "Beijing");
        assert!((location.latitude - 39.9075).abs() < f64::EPSILON);
    }

    #[test]
    fn missing_geocoding_result_is_none() {
        assert!(parse_location(&json!({})).is_none());
    }

    #[test]
    fn parses_forecast_summary() {
        let value = json!({
            "current": { "temperature_2m": 23.4, "weather_code": 61 }
        });
        let location = Location {
            latitude: 0.0,
            longitude: 0.0,
            label: "Beijing".into(),
        };
        let summary = parse_forecast(&location, &value).unwrap();
        assert!(!summary.contains("北京"));
        assert!(summary.contains("Beijing"));
        assert!(summary.contains("23.4"));
        assert!(summary.contains("雨"));
    }

    #[test]
    fn missing_current_is_none() {
        let location = Location {
            latitude: 0.0,
            longitude: 0.0,
            label: "Beijing".into(),
        };
        assert!(parse_forecast(&location, &json!({})).is_none());
    }

    #[tokio::test]
    async fn rejects_missing_city() {
        assert!(GetWeather::new().call(&json!({})).await.is_err());
    }

    #[tokio::test]
    async fn rejects_empty_city() {
        assert!(
            GetWeather::new()
                .call(&json!({ "city": "   " }))
                .await
                .is_err()
        );
    }

    #[test]
    fn spec_schema_matches_type() {
        let spec = GetWeather::new().spec();
        assert_eq!(spec.name, "get_weather");
        assert_eq!(spec.parameters["type"], "object");
        assert_eq!(spec.parameters["properties"]["city"]["type"], "string");
        assert_eq!(spec.parameters["required"], json!(["city"]));
        assert_eq!(spec.parameters["additionalProperties"], false);
    }
}
