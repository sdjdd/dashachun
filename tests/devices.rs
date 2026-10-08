mod common;

use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode};
use sqlx::PgPool;
use tower::ServiceExt;

use common::{body_json, post_json, register};
use dashachun::device::DeviceStore;

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

fn get_with_cookie(path: &str, cookie: String) -> Request<Body> {
    Request::builder()
        .uri(path)
        .header("cookie", cookie)
        .body(Body::empty())
        .unwrap()
}

fn delete_with_cookie(path: &str, cookie: String) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(path)
        .header("cookie", cookie)
        .body(Body::empty())
        .unwrap()
}

async fn create_agent(pool: &PgPool, user_id: i64) -> i64 {
    sqlx::query_scalar(
        "INSERT INTO agents (user_id, name, persona_prompt) VALUES ($1, 'default', '') RETURNING id",
    )
    .bind(user_id)
    .fetch_one(pool)
    .await
    .unwrap()
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn ota_creates_device_and_returns_code(pool: PgPool) {
    let app = common::app(pool);
    let res = app.clone().oneshot(ota_request()).await.unwrap();
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
    let res = app.clone().oneshot(ota_request()).await.unwrap();
    let body = body_json(res).await;
    assert_eq!(body["activation"]["code"].as_str(), Some(code));

    let res = app.clone().oneshot(activate_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::ACCEPTED);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn bind_flow_issues_token_and_gateway_enforces(pool: PgPool) {
    let app = common::app(pool.clone());
    let code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let (cookie, user_id) = register(&app, &pool).await;
    let agent_id = create_agent(&pool, user_id).await;

    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}","agent_id":999999}}"#),
            Some(cookie.clone()),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}","agent_id":{agent_id}}}"#),
            Some(cookie.clone()),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let device = body_json(res).await;
    assert_eq!(device["client_id"], CLIENT_ID);
    assert_eq!(device["agent_id"], agent_id);
    assert!(device.get("id").is_none());
    assert!(device["created_at"].as_str().unwrap().contains('T'));

    let res = app
        .clone()
        .oneshot(get_with_cookie("/api/devices", cookie.clone()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await.as_array().unwrap().len(), 1);

    assert_eq!(
        app.clone()
            .oneshot(activate_request())
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );

    let token = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        assert!(body.get("activation").is_none());
        body["websocket"]["token"].as_str().unwrap().to_owned()
    };
    assert_eq!(token.len(), 32);

    let state = common::test_state(pool.clone(), None);
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = dashachun::app(state, common::auth_state(pool.clone()));
    let server = tokio::spawn(dashachun::serve(
        listener,
        served,
        shutdown_rx,
        std::time::Duration::from_secs(5),
    ));

    assert!(connect_gateway(addr, Some(&token)).await.is_ok());
    assert!(connect_gateway(addr, Some("wrong-token")).await.is_err());
    assert!(connect_gateway(addr, None).await.is_err());

    sqlx::query("DELETE FROM agents WHERE id = $1")
        .bind(agent_id)
        .execute(&pool)
        .await
        .unwrap();
    assert!(
        connect_gateway(addr, Some(&token)).await.is_err(),
        "a bound device whose agent is gone must be rejected"
    );

    let res = app
        .clone()
        .oneshot(delete_with_cookie(
            &format!("/api/devices/{CLIENT_ID}"),
            cookie,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    assert!(connect_gateway(addr, Some(&token)).await.is_err());

    let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
    assert!(body["activation"]["code"].is_string());
    assert_eq!(body["websocket"]["token"], "");

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
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn expired_code_is_not_found_and_reissued(pool: PgPool) {
    let app = common::app(pool.clone());
    let old_code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let client_id = uuid::Uuid::parse_str(CLIENT_ID).unwrap();
    DeviceStore::new(pool.clone())
        .upsert(client_id, "aa:bb:cc:dd:ee:ff", "test-board")
        .await
        .unwrap();
    sqlx::query(
        "UPDATE devices SET activation_code_expires_at = now() - interval '1 minute' \
         WHERE client_id = $1",
    )
    .bind(client_id)
    .execute(&pool)
    .await
    .unwrap();

    let (cookie, user_id) = register(&app, &pool).await;
    let agent_id = create_agent(&pool, user_id).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{old_code}","agent_id":{agent_id}}}"#),
            Some(cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
    let new_code = body["activation"]["code"].as_str().unwrap();
    assert_ne!(new_code, old_code);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn binding_taken_code_is_not_found(pool: PgPool) {
    let app = common::app(pool.clone());
    let code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let (cookie, user_id) = register(&app, &pool).await;
    let agent_id = create_agent(&pool, user_id).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}","agent_id":{agent_id}}}"#),
            Some(cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (other, other_id) = register(&app, &pool).await;
    let other_agent = create_agent(&pool, other_id).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}","agent_id":{other_agent}}}"#),
            Some(other),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn binding_with_foreign_agent_is_rejected_and_leaves_code_usable(pool: PgPool) {
    let app = common::app(pool.clone());
    let code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let (owner, owner_id) = register(&app, &pool).await;
    let owner_agent = create_agent(&pool, owner_id).await;
    let (other, _) = register(&app, &pool).await;

    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}","agent_id":{owner_agent}}}"#),
            Some(other),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}","agent_id":{owner_agent}}}"#),
            Some(owner),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}
