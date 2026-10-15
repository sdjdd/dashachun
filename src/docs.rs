use axum::Json;
use utoipa::Modify;
use utoipa::OpenApi;
use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};

use crate::agent::dto::{AgentResponse, CreateAgentArgs};
use crate::auth::Role;
use crate::auth::dto::{ChangePasswordArgs, LoginArgs, RegisterArgs, UserResponse};
use crate::device::dto::{ActivateDeviceArgs, DeviceResponse};
use crate::error::ErrorResponse;
use crate::provider::dto::{CreateProviderArgs, ProviderResponse, UpdateProviderArgs};
use crate::settings::AudioCaptureSettings;
use crate::voiceprint::dto::{CreateVoiceprintArgs, UpdateVoiceprintArgs, VoiceprintResponse};

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
        crate::routes::auth::register,
        crate::routes::auth::login,
        crate::routes::auth::logout,
        crate::routes::auth::change_password,
        crate::routes::auth::me,
        crate::routes::agents::create_agent,
        crate::routes::devices::list_devices,
        crate::routes::devices::activate_device,
        crate::routes::devices::unbind_device,
        crate::routes::providers::list_providers,
        crate::routes::providers::create_provider,
        crate::routes::providers::get_provider,
        crate::routes::providers::update_provider,
        crate::routes::providers::delete_provider,
        crate::routes::providers::activate_provider,
        crate::routes::audio_capture::get_capture,
        crate::routes::audio_capture::put_capture,
        crate::routes::voiceprints::list_voiceprints,
        crate::routes::voiceprints::create_voiceprint,
        crate::routes::voiceprints::update_voiceprint,
        crate::routes::voiceprints::delete_voiceprint,
        crate::routes::messages::message_audio,
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
        CreateVoiceprintArgs,
        UpdateVoiceprintArgs,
        VoiceprintResponse,
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
            "/api/devices",
            "/api/devices/activate",
            "/api/devices/{client_id}",
            "/api/providers",
            "/api/providers/{id}",
            "/api/providers/{id}/activate",
            "/api/settings/audio-capture",
            "/api/agents/{agent_id}/voiceprints",
            "/api/agents/{agent_id}/voiceprints/{voiceprint_id}",
            "/api/agents/{agent_id}/messages/{message_id}/audio",
        ] {
            assert!(
                paths.contains(&expected.to_owned()),
                "missing {expected}: {paths:?}"
            );
        }
        assert_eq!(paths.len(), 16, "unexpected extra paths: {paths:?}");

        let json = serde_json::to_value(&doc).unwrap();
        assert!(json["openapi"].as_str().unwrap().starts_with("3.1"));
        assert!(json["paths"]["/api/devices/{client_id}"]["delete"]["parameters"].is_array());
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
