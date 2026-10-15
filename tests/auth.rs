mod common;

use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use axum_extra::extract::cookie::Key;
use serde_json::json;
use sqlx::PgPool;

use common::{app, body_json, expect_json, request, send, session_cookie};
use dashachun::http::routes::auth::LOGIN_LIMIT;
use dashachun::http::session::AuthState;

fn post_from(path: &str, body: &str, ip: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("x-forwarded-for", ip)
        .body(Body::from(body.to_owned()))
        .unwrap()
}

fn post_with(path: &str, body: &str, headers: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::from(body.to_owned())).unwrap()
}

fn get_with(path: &str, cookie: &str, headers: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder()
        .method("GET")
        .uri(path)
        .header("cookie", cookie);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::empty()).unwrap()
}

/// The trust-proxy auth state, shared by the tests that assert on the
/// captured client IP.
fn trust_proxy_auth(pool: PgPool) -> AuthState {
    AuthState {
        pool,
        key: Key::from(&[7u8; 64]),
        ttl_secs: 3600,
        cookie_name: "xz_session".into(),
        cookie_secure: false,
        limiter: common::test_limiter(),
        trust_proxy: true,
    }
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
    let auth = trust_proxy_auth(pool.clone());
    let app = dashachun::http::app(common::test_state(pool, None), auth);
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

#[sqlx::test]
#[ignore]
async fn login_records_the_client_ip_and_user_agent(pool: PgPool) {
    let app = dashachun::http::app(
        common::test_state(pool.clone(), None),
        trust_proxy_auth(pool.clone()),
    );
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
        post_with(
            "/api/auth/login",
            &json!({"username": username, "password": "supersecret1"}).to_string(),
            &[
                ("x-forwarded-for", "203.0.113.9, 198.51.100.7"),
                ("user-agent", "probe/1.0"),
            ],
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);

    let (ip, ua): (String, String) = sqlx::query_as(
        "SELECT ip, user_agent FROM auth_sessions \
         WHERE user_id = (SELECT id FROM users WHERE username = $1) \
         ORDER BY created_at DESC LIMIT 1",
    )
    .bind(&username)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(ip, "198.51.100.7");
    assert_eq!(ua, "probe/1.0");
}

#[sqlx::test]
#[ignore]
async fn activity_writes_are_throttled_within_the_interval(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = common::register(&app, &pool).await;
    async fn snapshot(pool: &PgPool, user_id: i64) -> (String, String, i64) {
        sqlx::query_as::<_, (String, String, i64)>(
            "SELECT ip, user_agent, EXTRACT(EPOCH FROM last_active_at)::BIGINT \
             FROM auth_sessions WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_one(pool)
        .await
        .unwrap()
    }
    let before = snapshot(&pool, user_id).await;

    for _ in 0..3 {
        expect_json(
            &app,
            request("GET", "/api/auth/me", None, Some(&cookie)),
            StatusCode::OK,
        )
        .await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;

    let after = snapshot(&pool, user_id).await;
    assert_eq!(after, before);
}

#[sqlx::test]
#[ignore]
async fn stale_activity_is_refreshed_with_the_latest_contact(pool: PgPool) {
    let app = dashachun::http::app(
        common::test_state(pool.clone(), None),
        trust_proxy_auth(pool.clone()),
    );
    let (cookie, user_id) = common::register(&app, &pool).await;
    sqlx::query(
        "UPDATE auth_sessions SET last_active_at = now() - interval '5 minutes' WHERE user_id = $1",
    )
    .bind(user_id)
    .execute(&pool)
    .await
    .unwrap();

    let res = send(
        &app,
        get_with(
            "/api/auth/me",
            &cookie,
            &[
                ("x-forwarded-for", "203.0.113.9, 198.51.100.7"),
                ("user-agent", "roaming/2.0"),
            ],
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);

    // The refresh runs off the response path — poll for it to land.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let (ip, ua): (String, String) =
            sqlx::query_as("SELECT ip, user_agent FROM auth_sessions WHERE user_id = $1")
                .bind(user_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        if ip == "198.51.100.7" {
            assert_eq!(ua, "roaming/2.0");
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "session was never refreshed, ip still {ip}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[sqlx::test]
#[ignore]
async fn sessions_list_scopes_and_revokes(pool: PgPool) {
    let app = app(pool.clone());
    let username = common::test_name();
    let creds = json!({"username": username, "password": "supersecret1"}).to_string();

    // One account, three live sessions: register plus two logins.
    let res = send(
        &app,
        request("POST", "/api/auth/register", Some(&creds), None),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
    let cookie1 = session_cookie(&res);
    let user_id: i64 = sqlx::query_scalar("SELECT id FROM users WHERE username = $1")
        .bind(&username)
        .fetch_one(&pool)
        .await
        .unwrap();
    let login = || async {
        let res = send(&app, request("POST", "/api/auth/login", Some(&creds), None)).await;
        assert_eq!(res.status(), StatusCode::OK);
        session_cookie(&res)
    };
    let _cookie2 = login().await;
    let cookie3 = login().await;

    // Name the sessions in the DB so the assertions can pick them
    // deterministically (the cookie is signed, so the raw value is not the
    // session id); an expired row joins to prove the list filters it.
    let ids: Vec<uuid::Uuid> =
        sqlx::query_as("SELECT id FROM auth_sessions WHERE user_id = $1 ORDER BY created_at")
            .bind(user_id)
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|(id,)| id)
            .collect();
    assert_eq!(ids.len(), 3);
    for (id, ua) in ids.iter().zip(["first", "second", "third"]) {
        sqlx::query("UPDATE auth_sessions SET user_agent = $2 WHERE id = $1")
            .bind(id)
            .bind(ua)
            .execute(&pool)
            .await
            .unwrap();
    }
    sqlx::query(
        "INSERT INTO auth_sessions (id, user_id, expires_at) \
         VALUES ($1, $2, now() - interval '1 hour')",
    )
    .bind(uuid::Uuid::new_v4())
    .bind(user_id)
    .execute(&pool)
    .await
    .unwrap();

    // Unauthenticated reads and revokes are rejected.
    let res = send(&app, request("GET", "/api/auth/sessions", None, None)).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/auth/sessions/{}", ids[0]),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // The list shows only the caller's live sessions; `current` marks the
    // session the request itself arrived on.
    let body = expect_json(
        &app,
        request("GET", "/api/auth/sessions", None, Some(&cookie1)),
        StatusCode::OK,
    )
    .await;
    let items = body.as_array().unwrap();
    assert_eq!(items.len(), 3);
    let by_ua = |ua: &str| {
        items
            .iter()
            .find(|item| item["user_agent"] == ua)
            .unwrap_or_else(|| panic!("no session with user-agent {ua} in {items:?}"))
    };
    assert_eq!(by_ua("first")["id"], ids[0].to_string());
    assert_eq!(by_ua("first")["current"], true);
    assert_eq!(by_ua("second")["current"], false);
    assert_eq!(by_ua("third")["current"], false);
    for item in items {
        assert!(item["ip"].is_string());
        assert!(item["created_at"].as_str().is_some());
        assert!(item["last_active_at"].as_str().is_some());
        assert!(item["expires_at"].as_str().is_some());
    }

    // Another account's session is not revocable and stays live.
    let (foreign_cookie, foreign_id) = common::register(&app, &pool).await;
    let foreign_session: (uuid::Uuid,) =
        sqlx::query_as("SELECT id FROM auth_sessions WHERE user_id = $1")
            .bind(foreign_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/auth/sessions/{}", foreign_session.0),
            None,
            Some(&cookie1),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request("GET", "/api/auth/me", None, Some(&foreign_cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);

    // Revoking one of the caller's other sessions kills exactly that
    // cookie.
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/auth/sessions/{}", ids[1]),
            None,
            Some(&cookie1),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    let res = send(&app, request("GET", "/api/auth/me", None, Some(&_cookie2))).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let body = expect_json(
        &app,
        request("GET", "/api/auth/sessions", None, Some(&cookie1)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body.as_array().unwrap().len(), 2);

    // Revoking the current session is a self-logout: the row dies and the
    // next request fails the lookup.
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/auth/sessions/{}", ids[0]),
            None,
            Some(&cookie1),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    let res = send(&app, request("GET", "/api/auth/me", None, Some(&cookie1))).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // The last session survives, still listed for its own cookie.
    let body = expect_json(
        &app,
        request("GET", "/api/auth/sessions", None, Some(&cookie3)),
        StatusCode::OK,
    )
    .await;
    let items = body.as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["id"], ids[2].to_string());
    assert_eq!(items[0]["current"], true);
}
