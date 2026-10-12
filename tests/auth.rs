mod common;

use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use axum_extra::extract::cookie::Key;
use serde_json::json;
use sqlx::PgPool;

use common::{app, body_json, expect_json, request, send, session_cookie};
use dashachun::auth::state::AuthState;
use dashachun::routes::auth::LOGIN_LIMIT;

fn post_from(path: &str, body: &str, ip: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("x-forwarded-for", ip)
        .body(Body::from(body.to_owned()))
        .unwrap()
}

#[sqlx::test]
#[ignore]
async fn register_login_me_change_logout(pool: PgPool) {
    let app = app(pool);
    let username = common::test_name();
    let password = "supersecret1";

    let res = send(
        &app,
        request(
            "POST",
            "/api/auth/register",
            Some(&json!({"username": username, "password": password}).to_string()),
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
    let cookie = session_cookie(&res);
    let body = body_json(res).await;
    assert_eq!(body["username"], username);
    assert!(body["id"].is_i64());
    assert_eq!(body["role"], "admin");

    let body = expect_json(
        &app,
        request("GET", "/api/auth/me", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["username"], username);
    assert_eq!(body["role"], "admin");

    let res = send(
        &app,
        request(
            "POST",
            "/api/auth/change-password",
            Some(&json!({"current_password": password, "new_password": "newsecret1"}).to_string()),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = send(
        &app,
        request(
            "POST",
            "/api/auth/login",
            Some(&json!({"username": username, "password": "newsecret1"}).to_string()),
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let cookie2 = session_cookie(&res);

    let res = send(
        &app,
        request("POST", "/api/auth/logout", None, Some(&cookie2)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
}

#[sqlx::test]
#[ignore]
async fn duplicate_username_conflicts(pool: PgPool) {
    let app = app(pool);
    let username = common::test_name();
    let body = json!({"username": username, "password": "supersecret1"}).to_string();
    let first = send(
        &app,
        request("POST", "/api/auth/register", Some(&body), None),
    )
    .await;
    assert_eq!(first.status(), StatusCode::CREATED);
    let second = send(
        &app,
        request("POST", "/api/auth/register", Some(&body), None),
    )
    .await;
    assert_eq!(second.status(), StatusCode::CONFLICT);
}

#[sqlx::test]
#[ignore]
async fn login_wrong_password_is_unauthorized(pool: PgPool) {
    let app = app(pool);
    let username = common::test_name();
    send(
        &app,
        request(
            "POST",
            "/api/auth/register",
            Some(&json!({"username": username, "password": "supersecret1"}).to_string()),
            None,
        ),
    )
    .await;

    let res = send(
        &app,
        request(
            "POST",
            "/api/auth/login",
            Some(&json!({"username": username, "password": "wrongpass1"}).to_string()),
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn invalid_register_is_bad_request(pool: PgPool) {
    let app = app(pool);
    let res = send(
        &app,
        request(
            "POST",
            "/api/auth/register",
            Some(r#"{"username":"ab","password":"short"}"#),
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore]
async fn me_without_cookie_is_unauthorized(pool: PgPool) {
    let app = app(pool);
    let res = send(&app, request("GET", "/api/auth/me", None, None)).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn repeated_logins_hit_the_rate_limit(pool: PgPool) {
    let app = app(pool);
    let body = r#"{"username":"ghost","password":"supersecret1"}"#;
    for _ in 0..LOGIN_LIMIT {
        let res = send(&app, request("POST", "/api/auth/login", Some(body), None)).await;
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }
    let res = send(&app, request("POST", "/api/auth/login", Some(body), None)).await;
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    let retry_after = res
        .headers()
        .get("retry-after")
        .unwrap()
        .to_str()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    assert!(retry_after >= 1);
}

#[sqlx::test]
#[ignore]
async fn the_rate_limit_is_per_client_ip(pool: PgPool) {
    let auth = AuthState {
        pool: pool.clone(),
        key: Key::from(&[7u8; 64]),
        ttl_secs: 3600,
        cookie_name: "xz_session".into(),
        cookie_secure: false,
        limiter: common::test_limiter(),
        trust_proxy: true,
    };
    let app = dashachun::app(common::test_state(pool, None), auth);
    let body = r#"{"username":"ghost","password":"supersecret1"}"#;
    for _ in 0..LOGIN_LIMIT {
        let res = send(&app, post_from("/api/auth/login", body, "203.0.113.7")).await;
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }
    let res = send(&app, post_from("/api/auth/login", body, "203.0.113.7")).await;
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    let res = send(&app, post_from("/api/auth/login", body, "203.0.113.8")).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn the_first_registration_is_admin_and_the_rest_are_not(pool: PgPool) {
    let app = app(pool.clone());
    let (first, _) = common::register(&app, &pool).await;
    let (second, _) = common::register(&app, &pool).await;

    let body =
        body_json(send(&app, request("GET", "/api/auth/me", None, Some(&first))).await).await;
    assert_eq!(body["role"], "admin");
    let body =
        body_json(send(&app, request("GET", "/api/auth/me", None, Some(&second))).await).await;
    assert_eq!(body["role"], "user");
}
