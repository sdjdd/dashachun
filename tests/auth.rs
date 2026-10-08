mod common;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use sqlx::PgPool;
use tower::ServiceExt;

use common::app;

fn post(path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::builder().uri(path).body(Body::empty()).unwrap()
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn register_login_me_change_logout(pool: PgPool) {
    let app = app(pool);
    let username = common::test_name();
    let password = "supersecret1";

    let res = app
        .clone()
        .oneshot(post(
            "/api/auth/register",
            &format!(r#"{{"username":"{username}","password":"{password}"}}"#),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let cookie = common::session_cookie(&res);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(res.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["username"], username);
    assert!(body["id"].is_i64());

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/auth/me")
                .header("cookie", cookie.clone())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(res.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["username"], username);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/change-password")
                .header("cookie", cookie.clone())
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    r#"{{"current_password":"{password}","new_password":"newsecret1"}}"#
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = app
        .clone()
        .oneshot(post(
            "/api/auth/login",
            &format!(r#"{{"username":"{username}","password":"newsecret1"}}"#),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let cookie2 = common::session_cookie(&res);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/logout")
                .header("cookie", cookie2)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn duplicate_username_conflicts(pool: PgPool) {
    let app = app(pool);
    let username = common::test_name();
    let body = format!(r#"{{"username":"{username}","password":"supersecret1"}}"#);
    let first = app
        .clone()
        .oneshot(post("/api/auth/register", &body))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::CREATED);
    let second = app
        .oneshot(post("/api/auth/register", &body))
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::CONFLICT);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn login_wrong_password_is_unauthorized(pool: PgPool) {
    let app = app(pool);
    let username = common::test_name();
    app.clone()
        .oneshot(post(
            "/api/auth/register",
            &format!(r#"{{"username":"{username}","password":"supersecret1"}}"#),
        ))
        .await
        .unwrap();

    let res = app
        .oneshot(post(
            "/api/auth/login",
            &format!(r#"{{"username":"{username}","password":"wrongpass1"}}"#),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn invalid_register_is_bad_request(pool: PgPool) {
    let app = app(pool);
    let res = app
        .oneshot(post(
            "/api/auth/register",
            r#"{"username":"ab","password":"short"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn me_without_cookie_is_unauthorized(pool: PgPool) {
    let app = app(pool);
    let res = app.oneshot(get("/api/auth/me")).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}
