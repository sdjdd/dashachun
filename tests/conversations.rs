//! The conversation transcript REST: the paged conversation and message
//! reads, their ownership scoping and pagination windows. The nested
//! audio playback route is exercised in the audio-capture suite.

mod common;

use axum::http::StatusCode;
use sqlx::PgPool;

use common::{expect_json, register, request, send};

use dashachun::agent::AgentStore;

async fn owned_agent(pool: &PgPool, user_id: i64, name: &str) -> i64 {
    AgentStore::new(pool.clone())
        .create(user_id, name, "")
        .await
        .unwrap()
        .id
}

async fn message_row(pool: &PgPool, conversation_id: i64, role: &str, text: &str) -> i64 {
    sqlx::query_scalar(
        "INSERT INTO messages (conversation_id, message) VALUES ($1, $2) RETURNING id",
    )
    .bind(conversation_id)
    .bind(serde_json::json!({ "role": role, "content": text }))
    .fetch_one(pool)
    .await
    .unwrap()
}

fn ids(body: &serde_json::Value) -> Vec<i64> {
    body["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_i64().unwrap())
        .collect()
}

#[sqlx::test]
#[ignore]
async fn conversations_list_scopes_and_paginates(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let (_other_cookie, other_user) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let other_agent = owned_agent(&pool, user_id, "b").await;
    let foreign_agent = owned_agent(&pool, other_user, "c").await;

    // Seeded oldest first, so the listing order is predictable.
    let first = common::conversation(&pool, 1, agent).await;
    let second = common::conversation(&pool, 1, agent).await;
    let third = common::conversation(&pool, 1, agent).await;
    let _other = common::conversation(&pool, 1, other_agent).await;
    let _foreign = common::conversation(&pool, 1, foreign_agent).await;

    // Unauthenticated reads are rejected.
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // A foreign agent and its conversations stay invisible.
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{foreign_agent}/conversations"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // The agent's own conversations, newest first, with the wire session
    // id recorded and nothing ended.
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(ids(&body), vec![third, second, first]);
    assert_eq!(body["has_more"], false);
    let item = &body["items"][0];
    assert!(item["session_id"].as_str().unwrap().len() >= 32);
    assert_eq!(item["device_id"], 1);
    assert!(item["created_at"].as_str().is_some());
    assert!(item["ended_at"].is_null());
    assert!(item.get("agent_id").is_none());

    // The caller's other agent is a different scope.
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{other_agent}/conversations"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["items"].as_array().unwrap().len(), 1);

    // A short page reports more; before_id continues where it ended.
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations?limit=2"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(ids(&body), vec![third, second]);
    assert_eq!(body["has_more"], true);
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations?limit=2&before_id={second}"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(ids(&body), vec![first]);
    assert_eq!(body["has_more"], false);

    // Invalid windows are rejected: zero, negative and over the cap.
    for query in ["limit=0", "limit=-1", "limit=201"] {
        let res = send(
            &app,
            request(
                "GET",
                &format!("/api/agents/{agent}/conversations?{query}"),
                None,
                Some(&cookie),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}

#[sqlx::test]
#[ignore]
async fn messages_list_scopes_and_paginates(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let other_agent = owned_agent(&pool, user_id, "b").await;

    let conv = common::conversation(&pool, 1, agent).await;
    let m1 = message_row(&pool, conv, "user", "你好").await;
    let m2 = message_row(&pool, conv, "assistant", "你好！").await;
    let m3 = message_row(&pool, conv, "tool", "{\"ok\":true}").await;
    sqlx::query("INSERT INTO messages (conversation_id, message, speaker_id) VALUES ($1, $2, $3)")
        .bind(conv)
        .bind(serde_json::json!({ "role": "user", "content": "我是谁" }))
        .bind(7_i64)
        .execute(&pool)
        .await
        .unwrap();
    let other_conv = common::conversation(&pool, 1, other_agent).await;
    let _other = message_row(&pool, other_conv, "user", "别处").await;

    // Unauthenticated reads are rejected.
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations/{conv}/messages"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Newest first, verbatim message JSON with the extracted role.
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations/{conv}/messages"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    let items = body["items"].as_array().unwrap();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0]["role"], "user");
    assert_eq!(items[0]["speaker_id"], 7);
    assert_eq!(items[0]["message"]["content"], "我是谁");
    assert_eq!(items[1]["id"], m3);
    assert_eq!(items[1]["role"], "tool");
    assert_eq!(items[1]["message"]["content"], "{\"ok\":true}");
    assert_eq!(items[1]["speaker_id"], serde_json::Value::Null);
    assert_eq!(items[2]["id"], m2);
    assert_eq!(items[3]["id"], m1);
    assert!(items[0]["created_at"].as_str().is_some());
    assert_eq!(body["has_more"], false);

    // A conversation of the caller's other agent is out of scope even
    // though both agents belong to the caller.
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations/{other_conv}/messages"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // Unknown conversation.
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations/999999/messages"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // Pagination shares the conversations convention.
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations/{conv}/messages?limit=3"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(ids(&body).len(), 3);
    assert_eq!(body["has_more"], true);
    let before = ids(&body)[2];
    let body = expect_json(
        &app,
        request(
            "GET",
            &format!(
                "/api/agents/{agent}/conversations/{conv}/messages?limit=3&before_id={before}"
            ),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(ids(&body), vec![m1]);
    assert_eq!(body["has_more"], false);

    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/conversations/{conv}/messages?limit=201"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}
