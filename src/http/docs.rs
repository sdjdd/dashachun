use axum::Json;
use utoipa::Modify;
use utoipa::OpenApi;
use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};

use crate::auth::Role;
use crate::error::ErrorResponse;
use crate::http::dto::agent::{AgentResponse, CreateAgentArgs};
use crate::http::dto::auth::{ChangePasswordArgs, LoginArgs, RegisterArgs, UserResponse};
use crate::http::dto::conversation::{ConversationPage, MessagePage};
use crate::http::dto::device::{ActivateDeviceArgs, DeviceResponse};
use crate::http::dto::provider::{CreateProviderArgs, ProviderResponse, UpdateProviderArgs};
use crate::http::dto::speaker::{CreateSpeakerArgs, SpeakerResponse, UpdateSpeakerArgs};
use crate::settings::AudioCaptureSettings;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "dashachun",
        version = env!("CARGO_PKG_VERSION"),
        description = "REST API of the dashachun voice-assistant server. Session-cookie \
                       authenticated: the cookie is HttpOnly, so authorize by calling \
                       POST /api/auth/login first.",
    ),
    paths(
        crate::http::routes::auth::register,
        crate::http::routes::auth::login,
        crate::http::routes::auth::logout,
        crate::http::routes::auth::change_password,
        crate::http::routes::auth::me,
        crate::http::routes::agents::create_agent,
        crate::http::routes::devices::list_devices,
        crate::http::routes::devices::activate_device,
        crate::http::routes::devices::unbind_device,
        crate::http::routes::providers::list_providers,
        crate::http::routes::providers::create_provider,
        crate::http::routes::providers::get_provider,
        crate::http::routes::providers::update_provider,
        crate::http::routes::providers::delete_provider,
        crate::http::routes::providers::activate_provider,
        crate::http::routes::audio_capture::get_capture,
        crate::http::routes::audio_capture::put_capture,
        crate::http::routes::speakers::list_speakers,
        crate::http::routes::speakers::create_speaker,
        crate::http::routes::speakers::update_speaker,
        crate::http::routes::speakers::delete_speaker,
        crate::http::routes::conversations::list_conversations,
        crate::http::routes::conversations::list_messages,
        crate::http::routes::conversations::message_audio,
    ),
    components(schemas(
        ErrorResponse,
        Role,
        RegisterArgs,
        LoginArgs,
        ChangePasswordArgs,
        UserResponse,
        CreateAgentArgs,
        AgentResponse,
        ActivateDeviceArgs,
        DeviceResponse,
        CreateProviderArgs,
        UpdateProviderArgs,
        ProviderResponse,
        AudioCaptureSettings,
        CreateSpeakerArgs,
        UpdateSpeakerArgs,
        SpeakerResponse,
        ConversationPage,
        MessagePage,
    )),
    modifiers(&CookieAuthAddon),
)]
pub struct ApiDoc;

struct CookieAuthAddon;

impl Modify for CookieAuthAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "cookieAuth",
            SecurityScheme::ApiKey(ApiKey::Cookie(ApiKeyValue::with_description(
                "xz_session",
                "Session cookie; the name is the SESSION_COOKIE_NAME default",
            ))),
        );
    }
}

pub async fn spec() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_lists_every_rest_route() {
        let doc = ApiDoc::openapi();
        let paths: Vec<String> = doc.paths.paths.keys().cloned().collect();
        for expected in [
            "/api/auth/register",
            "/api/auth/login",
            "/api/auth/logout",
            "/api/auth/change-password",
            "/api/auth/me",
            "/api/agents",
            "/api/agents/{agent_id}/devices",
            "/api/agents/{agent_id}/devices/{client_id}",
            "/api/providers",
            "/api/providers/{id}",
            "/api/providers/{id}/activate",
            "/api/settings/audio-capture",
            "/api/agents/{agent_id}/speakers",
            "/api/agents/{agent_id}/speakers/{speaker_id}",
            "/api/agents/{agent_id}/conversations",
            "/api/agents/{agent_id}/conversations/{conversation_id}/messages",
            "/api/agents/{agent_id}/conversations/{conversation_id}/messages/{message_id}/audio",
        ] {
            assert!(
                paths.contains(&expected.to_owned()),
                "missing {expected}: {paths:?}"
            );
        }
        assert_eq!(paths.len(), 17, "unexpected extra paths: {paths:?}");

        let json = serde_json::to_value(&doc).unwrap();
        assert!(json["openapi"].as_str().unwrap().starts_with("3.1"));
        assert!(
            json["paths"]["/api/agents/{agent_id}/devices/{client_id}"]["delete"]["parameters"]
                .is_array()
        );
        assert!(json["components"]["schemas"]["ErrorResponse"].is_object());
        assert_eq!(
            json["components"]["securitySchemes"]["cookieAuth"]["type"],
            "apiKey"
        );
        assert_eq!(
            json["paths"]["/api/auth/login"]["post"]["responses"]["401"]["content"]["application/json"]
                ["schema"]["$ref"],
            "#/components/schemas/ErrorResponse"
        );
    }
}
