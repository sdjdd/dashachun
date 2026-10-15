mod common;

use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode};
use serde_json::json;
use sqlx::PgPool;

use common::{body_json, register, request, send};

const CLIENT_ID: &str = "11111111-1111-4111-8111-111111111111";

fn ota_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/api/ota")
        .header("content-type", "application/json")
        .header("host", "127.0.0.1:3000")
        .header("device-id", "aa:bb:cc:dd:ee:ff")
        .header("client-id", CLIENT_ID)
        .body(Body::from(
            r#"{"application":{"version":"1.0.0"},"board":{"type":"test-board"}}"#,
        ))
        .unwrap()
}

fn activate_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/api/ota/activate")
        .header("client-id", CLIENT_ID)
        .body(Body::from("{}"))
        .unwrap()
}

async fn create_agent(pool: &PgPool, user_id: i64, name: &str) -> i64 {
    sqlx::query_scalar(
        "INSERT INTO agents (user_id, name, persona_prompt) VALUES ($1, $2, '') RETURNING id",
    )
    .bind(user_id)
    .bind(name)
    .fetch_one(pool)
    .await
    .unwrap()
}

#[sqlx::test]
#[ignore]
async fn ota_creates_device_and_returns_code(pool: PgPool) {
    let app = common::app(pool);
    let res = send(&app, ota_request()).await;
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_json(res).await;
    let code = body["activation"]["code"]
        .as_str()
        .expect("activation code");
    assert_eq!(code.len(), 6);
    assert!(code.chars().all(|c| c.is_ascii_digit()));
    assert_eq!(body["activation"]["message"], code);
    assert!(body["activation"]["challenge"].is_string());
    assert_eq!(body["websocket"]["token"], "");

    // the same device re-polls OTA: the code is reused, not rotated
    let res = send(&app, ota_request()).await;
    let body = body_json(res).await;
    assert_eq!(body["activation"]["code"].as_str(), Some(code));

    let res = send(&app, activate_request()).await;
    assert_eq!(res.status(), StatusCode::ACCEPTED);
}

#[sqlx::test]
#[ignore]
async fn bind_flow_issues_token_and_gateway_enforces(pool: PgPool) {
    let app = common::app(pool.clone());
    let code = {
        let body = body_json(send(&app, ota_request()).await).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let (cookie, user_id) = register(&app, &pool).await;
    let agent_id = create_agent(&pool, user_id, "default").await;

    let res = send(
        &app,
        request(
            "POST",
            "/api/agents/999999/devices",
            Some(&json!({"code": &code}).to_string()),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent_id}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let device = body_json(res).await;
    assert_eq!(device["client_id"], CLIENT_ID);
    assert_eq!(device["agent_id"], agent_id);
    assert!(device.get("id").is_none());
    assert!(device["created_at"].as_str().unwrap().contains('T'));

    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent_id}/devices"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await.as_array().unwrap().len(), 1);

    assert_eq!(
        send(&app, activate_request()).await.status(),
        StatusCode::OK
    );

    let token = {
        let body = body_json(send(&app, ota_request()).await).await;
        assert!(body.get("activation").is_none());
        body["websocket"]["token"].as_str().unwrap().to_owned()
    };
    assert_eq!(token.len(), 32);

    let state = common::test_state(pool.clone(), None);
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = dashachun::http::app(state, common::auth_state(pool.clone()));
    let server = tokio::spawn(dashachun::serve(
        listener,
        served,
        shutdown_rx,
        std::time::Duration::from_secs(5),
    ));

    assert!(connect_gateway(addr, Some(&token)).await.is_ok());
    assert!(connect_gateway(addr, Some("wrong-token")).await.is_err());
    assert!(connect_gateway(addr, None).await.is_err());

    // the unbind is agent-scoped, so it must happen while the agent row
    // still exists
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{agent_id}/devices/{CLIENT_ID}"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    assert!(connect_gateway(addr, Some(&token)).await.is_err());

    let body = body_json(send(&app, ota_request()).await).await;
    assert!(body["activation"]["code"].is_string());
    assert_eq!(body["websocket"]["token"], "");

    // the freed code re-binds under the same agent, and a device whose
    // agent row is deleted afterwards is still rejected at the gateway
    let code = body["activation"]["code"].as_str().unwrap().to_owned();
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent_id}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        send(&app, activate_request()).await.status(),
        StatusCode::OK
    );
    let token = {
        let body = body_json(send(&app, ota_request()).await).await;
        assert!(body.get("activation").is_none());
        body["websocket"]["token"].as_str().unwrap().to_owned()
    };
    assert!(connect_gateway(addr, Some(&token)).await.is_ok());

    sqlx::query("DELETE FROM agents WHERE id = $1")
        .bind(agent_id)
        .execute(&pool)
        .await
        .unwrap();
    assert!(
        connect_gateway(addr, Some(&token)).await.is_err(),
        "a bound device whose agent is gone must be rejected"
    );

    shutdown_tx.send(true).unwrap();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server).await;
}

async fn connect_gateway(
    addr: std::net::SocketAddr,
    token: Option<&str>,
) -> Result<(), tokio_tungstenite::tungstenite::Error> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let mut request = format!("ws://{addr}/gateway")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("client-id", HeaderValue::from_static(CLIENT_ID));
    if let Some(token) = token {
        request.headers_mut().insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
    }
    let (_ws, _) = tokio_tungstenite::connect_async(request).await?;
    Ok(())
}

#[sqlx::test]
#[ignore]
async fn expired_code_is_not_found_and_reissued(pool: PgPool) {
    let (devices, prefix) = common::test_device_store(pool.clone());
    let app = dashachun::http::app(
        common::test_state_with_devices(pool.clone(), None, devices),
        common::auth_state(pool.clone()),
    );
    let old_code = {
        let body = body_json(send(&app, ota_request()).await).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let client_id = uuid::Uuid::parse_str(CLIENT_ID).unwrap();
    // simulate expiry: both keys of the issued code are gone
    let url = std::env::var("REDIS_URL").unwrap();
    let mut conn = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    for key in [
        format!("{prefix}:device:{client_id}"),
        format!("{prefix}:code:{old_code}"),
    ] {
        redis::cmd("DEL")
            .arg(key)
            .query_async::<()>(&mut conn)
            .await
            .unwrap();
    }

    let (cookie, user_id) = register(&app, &pool).await;
    let agent_id = create_agent(&pool, user_id, "default").await;
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent_id}/devices"),
            Some(&json!({"code": &old_code}).to_string()),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let body = body_json(send(&app, ota_request()).await).await;
    let new_code = body["activation"]["code"].as_str().unwrap();
    assert_ne!(new_code, old_code);
}

#[sqlx::test]
#[ignore]
async fn binding_taken_code_is_not_found(pool: PgPool) {
    let app = common::app(pool.clone());
    let code = {
        let body = body_json(send(&app, ota_request()).await).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let (cookie, user_id) = register(&app, &pool).await;
    let agent_id = create_agent(&pool, user_id, "default").await;
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent_id}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);

    let (other, other_id) = register(&app, &pool).await;
    let other_agent = create_agent(&pool, other_id, "other").await;
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{other_agent}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore]
async fn binding_with_foreign_agent_is_rejected_and_leaves_code_usable(pool: PgPool) {
    let app = common::app(pool.clone());
    let code = {
        let body = body_json(send(&app, ota_request()).await).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let (owner, owner_id) = register(&app, &pool).await;
    let owner_agent = create_agent(&pool, owner_id, "default").await;
    let (other, _) = register(&app, &pool).await;

    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{owner_agent}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{owner_agent}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
}

#[sqlx::test]
#[ignore]
async fn device_rest_is_scoped_to_the_path_agent(pool: PgPool) {
    let app = common::app(pool.clone());
    let (owner, owner_id) = register(&app, &pool).await;
    let bound_agent = create_agent(&pool, owner_id, "bound").await;
    let other_agent = create_agent(&pool, owner_id, "other").await;
    let (foreign, _) = register(&app, &pool).await;

    let code = {
        let body = body_json(send(&app, ota_request()).await).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{bound_agent}/devices"),
            Some(&json!({"code": &code}).to_string()),
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);

    // a foreign agent is 404 before anything else is looked at
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{bound_agent}/devices"),
            None,
            Some(&foreign),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // the device shows under its own agent, not the owner's other one
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{bound_agent}/devices"),
            None,
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let devices = body_json(res).await;
    assert_eq!(devices.as_array().unwrap().len(), 1);
    assert_eq!(devices[0]["client_id"], CLIENT_ID);

    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{other_agent}/devices"),
            None,
            Some(&owner),
        ),
    )
    .await;
    let devices = body_json(res).await;
    assert_eq!(devices.as_array().unwrap().len(), 0);

    // unbinding under the wrong agent leaves the binding untouched
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{other_agent}/devices/{CLIENT_ID}"),
            None,
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{bound_agent}/devices"),
            None,
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(body_json(res).await.as_array().unwrap().len(), 1);

    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{bound_agent}/devices/{CLIENT_ID}"),
            None,
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{bound_agent}/devices"),
            None,
            Some(&owner),
        ),
    )
    .await;
    assert_eq!(body_json(res).await.as_array().unwrap().len(), 0);
}
