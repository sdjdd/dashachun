mod common;

use axum::http::StatusCode;
use sqlx::PgPool;
use uuid::Uuid;

use common::{expect_json, register, request, send};

#[sqlx::test]
#[ignore]
async fn list_agents_scopes_to_the_caller_in_creation_order(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let body = expect_json(
        &app,
        request("GET", "/api/agents", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body.as_array().unwrap().len(), 0);

    for name in ["first", "second"] {
        send(
            &app,
            request(
                "POST",
                "/api/agents",
                Some(&format!(r#"{{"name":"{name}"}}"#)),
                Some(&cookie),
            ),
        )
        .await;
    }

    let (other, _) = register(&app, &pool).await;
    send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"first"}"#),
            Some(&other),
        ),
    )
    .await;

    let body = expect_json(
        &app,
        request("GET", "/api/agents", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    let names: Vec<&str> = body
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["first", "second"]);
    for agent in body.as_array().unwrap() {
        assert!(agent["id"].is_i64());
        assert!(agent["persona_prompt"].is_string());
        assert!(agent["created_at"].is_string());
    }

    let body = expect_json(
        &app,
        request("GET", "/api/agents", None, Some(&other)),
        StatusCode::OK,
    )
    .await;
    let names: Vec<&str> = body
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["first"]);
}

#[sqlx::test]
#[ignore]
async fn list_agents_requires_session(pool: PgPool) {
    let app = common::app(pool.clone());
    let res = send(&app, request("GET", "/api/agents", None, None)).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn create_agent_returns_created_agent(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let body = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default","persona_prompt":"你是一个测试助手"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(body["name"], "default");
    assert_eq!(body["persona_prompt"], "你是一个测试助手");
    assert!(body["id"].is_i64());
    assert!(body["created_at"].is_string());
}

#[sqlx::test]
#[ignore]
async fn create_agent_defaults_persona_prompt_to_empty(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let body = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(body["persona_prompt"], "");
}

#[sqlx::test]
#[ignore]
async fn get_agent_scopes_to_the_owner(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let created = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default","persona_prompt":"你是一个测试助手"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    let agent = created["id"].as_i64().unwrap();

    let body = expect_json(
        &app,
        request("GET", &format!("/api/agents/{agent}"), None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["name"], "default");
    assert_eq!(body["persona_prompt"], "你是一个测试助手");
    assert!(body["created_at"].is_string());

    let (other, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request("GET", &format!("/api/agents/{agent}"), None, Some(&other)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request("GET", "/api/agents/99999", None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request("GET", &format!("/api/agents/{agent}"), None, None),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn duplicate_agent_name_conflicts_within_user_but_not_across_users(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
    let res = send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CONFLICT);
    let (other, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
}

#[sqlx::test]
#[ignore]
async fn create_agent_rejects_blank_name(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request("POST", "/api/agents", Some(r#"{"name":""}"#), Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore]
async fn create_agent_requires_session(pool: PgPool) {
    let app = common::app(pool.clone());
    let res = send(
        &app,
        request("POST", "/api/agents", Some(r#"{"name":"default"}"#), None),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn update_agent_edits_fields_and_enforces_ownership(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let created = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default","persona_prompt":"旧的"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    let agent = created["id"].as_i64().unwrap();

    let body = expect_json(
        &app,
        request(
            "PUT",
            &format!("/api/agents/{agent}"),
            Some(r#"{"name":"renamed","persona_prompt":"新的"}"#),
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["name"], "renamed");
    assert_eq!(body["persona_prompt"], "新的");

    let body = expect_json(
        &app,
        request(
            "PUT",
            &format!("/api/agents/{agent}"),
            Some(r#"{"persona_prompt":"第三版"}"#),
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["name"], "renamed");
    assert_eq!(body["persona_prompt"], "第三版");

    let res = send(
        &app,
        request(
            "PUT",
            &format!("/api/agents/{agent}"),
            Some("{}"),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let (other, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "PUT",
            &format!("/api/agents/{agent}"),
            Some(r#"{"name":"hijack"}"#),
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = expect_json(
        &app,
        request("GET", "/api/agents", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body[0]["name"], "renamed");
}

#[sqlx::test]
#[ignore]
async fn update_agent_conflicts_on_a_taken_name(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    for name in ["a", "b"] {
        send(
            &app,
            request(
                "POST",
                "/api/agents",
                Some(&format!(r#"{{"name":"{name}"}}"#)),
                Some(&cookie),
            ),
        )
        .await;
    }
    let agents = expect_json(
        &app,
        request("GET", "/api/agents", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    let b = agents[1]["id"].as_i64().unwrap();
    let res = send(
        &app,
        request(
            "PUT",
            &format!("/api/agents/{b}"),
            Some(r#"{"name":"a"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CONFLICT);
}

#[sqlx::test]
#[ignore]
async fn delete_agent_removes_sub_resources_and_unbinds_devices(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let doomed = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"doomed"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await["id"]
        .as_i64()
        .unwrap();
    let keeper = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"keeper"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await["id"]
        .as_i64()
        .unwrap();

    let device_a: i64 = sqlx::query_scalar(
        "INSERT INTO devices (client_id, user_id, agent_id, token_hash, activated_at) \
         VALUES ($1, $2, $3, 'hash-a', now()) RETURNING id",
    )
    .bind(Uuid::new_v4())
    .bind(user_id)
    .bind(doomed)
    .fetch_one(&pool)
    .await
    .unwrap();
    let device_b: i64 = sqlx::query_scalar(
        "INSERT INTO devices (client_id, user_id, agent_id, token_hash, activated_at) \
         VALUES ($1, $2, $3, 'hash-b', now()) RETURNING id",
    )
    .bind(Uuid::new_v4())
    .bind(user_id)
    .bind(keeper)
    .fetch_one(&pool)
    .await
    .unwrap();

    let conv_a1 = common::conversation(&pool, device_a, doomed).await;
    let conv_a2 = common::conversation(&pool, device_a, doomed).await;
    let conv_b = common::conversation(&pool, device_b, keeper).await;
    let message = serde_json::json!({"role": "user", "content": "你好"});
    let msg_a: i64 = sqlx::query_scalar(
        "INSERT INTO messages (conversation_id, message) VALUES ($1, $2) RETURNING id",
    )
    .bind(conv_a1)
    .bind(&message)
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO messages (conversation_id, message) VALUES ($1, $2)")
        .bind(conv_b)
        .bind(&message)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO message_audios (message_id, bucket, object_key, sample_rate, channels, duration_ms) \
         VALUES ($1, 'bucket-a', 'key-a', 16000, 1, 1000)",
    )
    .bind(msg_a)
    .execute(&pool)
    .await
    .unwrap();
    for (agent, name) in [(doomed, "甲"), (keeper, "乙")] {
        sqlx::query(
            "INSERT INTO speakers (agent_id, name, embedding, sample_count) \
             VALUES ($1, $2, array_fill(0.1::real, ARRAY[192]), 1)",
        )
        .bind(agent)
        .bind(name)
        .execute(&pool)
        .await
        .unwrap();
    }
    for agent in [doomed, keeper] {
        sqlx::query(
            "INSERT INTO memory_entries (user_id, agent_id, mem_no, content) \
             VALUES ($1, $2, 1, '一条记忆')",
        )
        .bind(user_id)
        .bind(agent)
        .execute(&pool)
        .await
        .unwrap();
    }

    let (other, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{doomed}"),
            None,
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{doomed}"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let names: Vec<String> = sqlx::query_scalar("SELECT name FROM agents ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();
    assert_eq!(names, vec!["keeper".to_owned()]);

    let (user_left, agent_left, token_left): (Option<i64>, Option<i64>, Option<String>) =
        sqlx::query_as("SELECT user_id, agent_id, token_hash FROM devices WHERE id = $1")
            .bind(device_a)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(user_left, None);
    assert_eq!(agent_left, None);
    assert_eq!(token_left, None);
    let (user_kept, agent_kept, token_kept): (Option<i64>, Option<i64>, Option<String>) =
        sqlx::query_as("SELECT user_id, agent_id, token_hash FROM devices WHERE id = $1")
            .bind(device_b)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(user_kept, Some(user_id));
    assert_eq!(agent_kept, Some(keeper));
    assert_eq!(token_kept.as_deref(), Some("hash-b"));

    async fn count(pool: &PgPool, sql: &str, bind: i64) -> i64 {
        sqlx::query_scalar(sql)
            .bind(bind)
            .fetch_one(pool)
            .await
            .unwrap()
    }
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM conversations WHERE agent_id = $1",
            doomed
        )
        .await,
        0
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM conversations WHERE agent_id = $1",
            keeper
        )
        .await,
        1
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM messages WHERE conversation_id = $1",
            conv_a1
        )
        .await,
        0
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM messages WHERE conversation_id = $1",
            conv_a2
        )
        .await,
        0
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM messages WHERE conversation_id = $1",
            conv_b
        )
        .await,
        1
    );
    let audios: i64 = sqlx::query_scalar("SELECT count(*) FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(audios, 0);
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM speakers WHERE agent_id = $1",
            doomed
        )
        .await,
        0
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM speakers WHERE agent_id = $1",
            keeper
        )
        .await,
        1
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM memory_entries WHERE agent_id = $1",
            doomed
        )
        .await,
        0
    );
    assert_eq!(
        count(
            &pool,
            "SELECT count(*) FROM memory_entries WHERE agent_id = $1",
            keeper
        )
        .await,
        1
    );

    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{doomed}"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{doomed}/conversations"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = expect_json(
        &app,
        request("GET", "/api/agents", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body[0]["name"], "keeper");
}
