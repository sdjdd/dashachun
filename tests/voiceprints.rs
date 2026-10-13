//! The voiceprints suite: store CRUD, ownership scoping, and the
//! element-wise append semantics, plus the REST surface over the same
//! store. Run with `cargo test --test voiceprints -- --ignored`.

mod common;

use axum::http::StatusCode;
use sqlx::PgPool;

use common::{app, expect_json, register, request, send};

use dashachun::agent::AgentStore;
use dashachun::error::AppError;
use dashachun::voiceprint::VoiceprintStore;

fn vec192(value: f32) -> Vec<f32> {
    vec![value; 192]
}

async fn owned_agent(pool: &PgPool, user_id: i64, name: &str) -> i64 {
    AgentStore::new(pool.clone())
        .create(user_id, name, "")
        .await
        .unwrap()
        .id
}

async fn seed(pool: &PgPool, agent_id: i64, user_id: i64, label: &str) -> i64 {
    VoiceprintStore::new(pool.clone())
        .create_or_append(agent_id, user_id, label, None, vec192(0.5), 1)
        .await
        .unwrap()
        .id
}

#[sqlx::test]
#[ignore]
async fn append_sums_elementwise_and_accumulates_sample_count(pool: PgPool) {
    let store = VoiceprintStore::new(pool.clone());
    let agent = owned_agent(&pool, 1, "a").await;
    let first = store
        .create_or_append(agent, 1, "张三", Some("主人"), vec192(0.5), 2)
        .await
        .unwrap();
    assert_eq!(first.sample_count, 2);
    assert_eq!(first.description, "主人");

    let second = store
        .create_or_append(agent, 1, "张三", None, vec192(0.25), 3)
        .await
        .unwrap();
    assert_eq!(second.id, first.id);
    assert_eq!(second.sample_count, 5);
    assert_eq!(second.description, "主人");

    let embedding: Vec<f32> = sqlx::query_scalar("SELECT embedding FROM voiceprints WHERE id = $1")
        .bind(first.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(embedding.iter().all(|&x| x == 0.75));
}

#[sqlx::test]
#[ignore]
async fn create_or_append_on_an_agent_you_do_not_own_is_not_found(pool: PgPool) {
    let store = VoiceprintStore::new(pool.clone());
    let agent = owned_agent(&pool, 2, "a").await;
    let cross_user = store
        .create_or_append(agent, 1, "张三", None, vec192(0.5), 1)
        .await
        .unwrap_err();
    assert!(matches!(cross_user, AppError::NotFound));
    let missing = store
        .create_or_append(999_999, 1, "张三", None, vec192(0.5), 1)
        .await
        .unwrap_err();
    assert!(matches!(missing, AppError::NotFound));
}

#[sqlx::test]
#[ignore]
async fn same_label_coexists_across_agents_and_lists_are_scoped(pool: PgPool) {
    let store = VoiceprintStore::new(pool.clone());
    let mine = owned_agent(&pool, 1, "mine").await;
    let theirs = owned_agent(&pool, 2, "theirs").await;
    store
        .create_or_append(mine, 1, "张三", None, vec192(0.5), 1)
        .await
        .unwrap();
    store
        .create_or_append(theirs, 2, "张三", None, vec192(0.5), 1)
        .await
        .unwrap();

    let mine_list = store.list_for_agent(mine, 1).await.unwrap();
    assert_eq!(mine_list.len(), 1);
    let other = store.get(mine, mine_list[0].id, 2).await.unwrap();
    assert!(other.is_none());
    assert!(store.list_for_agent(mine, 2).await.unwrap().is_empty());
    assert!(
        store
            .get(theirs, mine_list[0].id, 2)
            .await
            .unwrap()
            .is_none()
    );
}

#[sqlx::test]
#[ignore]
async fn update_renames_and_conflicts_on_a_taken_label(pool: PgPool) {
    let store = VoiceprintStore::new(pool.clone());
    let agent = owned_agent(&pool, 1, "a").await;
    let first = store
        .create_or_append(agent, 1, "a-label", None, vec192(0.5), 1)
        .await
        .unwrap();
    store
        .create_or_append(agent, 1, "b-label", None, vec192(0.5), 1)
        .await
        .unwrap();

    let renamed = store
        .update(agent, first.id, 1, Some("c-label"), Some("新描述"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(renamed.label, "c-label");
    assert_eq!(renamed.description, "新描述");

    let err = store
        .update(agent, first.id, 1, Some("b-label"), None)
        .await
        .unwrap_err();
    assert!(matches!(err, AppError::Conflict(_)));

    let kept = store
        .update(agent, first.id, 1, None, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(kept.label, "c-label");
    assert_eq!(kept.description, "新描述");

    let foreign = store
        .update(agent, first.id, 2, Some("stolen"), None)
        .await
        .unwrap();
    assert!(foreign.is_none());
}

#[sqlx::test]
#[ignore]
async fn delete_is_scoped_by_ownership(pool: PgPool) {
    let store = VoiceprintStore::new(pool.clone());
    let agent = owned_agent(&pool, 1, "a").await;
    let record = store
        .create_or_append(agent, 1, "张三", None, vec192(0.5), 1)
        .await
        .unwrap();

    assert!(!store.delete(agent, record.id, 2).await.unwrap());
    assert!(store.delete(agent, record.id, 1).await.unwrap());
    assert!(!store.delete(agent, record.id, 1).await.unwrap());
    assert!(store.list_for_agent(agent, 1).await.unwrap().is_empty());
}

#[sqlx::test]
#[ignore]
async fn list_scopes_by_ownership_and_hides_the_embedding(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    seed(&pool, agent, user_id, "张三").await;

    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/voiceprints"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body.as_array().unwrap().len(), 1);
    let entry = &body[0];
    assert_eq!(entry["label"], "张三");
    assert_eq!(entry["agent_id"], agent);
    assert_eq!(entry["sample_count"], 1);
    assert!(entry["embedding"].is_null());
    assert!(entry["created_at"].is_string());
    assert!(entry["updated_at"].is_string());

    let (other_cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/voiceprints"),
            None,
            Some(&other_cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request("GET", "/api/agents/999999/voiceprints", None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/voiceprints"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn create_rejects_invalid_payloads_and_503s_without_the_model(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let path = format!("/api/agents/{agent}/voiceprints");

    for payload in [
        r#"{}"#,
        r#"{"label":"","message_ids":[1]}"#,
        r#"{"label":"张三","message_ids":[]}"#,
        r#"{"label":"张三","message_ids":[1,2,3,4,5,6,7,8,9,10,11]}"#,
    ] {
        let res = send(&app, request("POST", &path, Some(payload), Some(&cookie))).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "payload: {payload}");
    }

    let res = send(
        &app,
        request(
            "POST",
            &path,
            Some(r#"{"label":"张三","description":"主人","message_ids":[1,2]}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[sqlx::test]
#[ignore]
async fn update_and_delete_over_rest_are_scoped_by_ownership(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let first = seed(&pool, agent, user_id, "first").await;
    seed(&pool, agent, user_id, "second").await;
    let path = format!("/api/agents/{agent}/voiceprints/{first}");

    let body = expect_json(
        &app,
        request(
            "PUT",
            &path,
            Some(r#"{"description":"新的描述"}"#),
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["label"], "first");
    assert_eq!(body["description"], "新的描述");

    for payload in [
        r#"{}"#,
        r#"{"label":""}"#,
        &format!(r#"{{"description":"{}"}}"#, "长".repeat(201)),
    ] {
        let res = send(&app, request("PUT", &path, Some(payload), Some(&cookie))).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "payload: {payload}");
    }

    let res = send(
        &app,
        request("PUT", &path, Some(r#"{"label":"second"}"#), Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CONFLICT);

    let (other_cookie, _) = register(&app, &pool).await;
    for (method, payload) in [("PUT", Some(r#"{"label":"stolen"}"#)), ("DELETE", None)] {
        let res = send(&app, request(method, &path, payload, Some(&other_cookie))).await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND, "{method}");
    }

    let res = send(&app, request("DELETE", &path, None, Some(&cookie))).await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    let res = send(&app, request("DELETE", &path, None, Some(&cookie))).await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request("PUT", &path, Some(r#"{"label":"third"}"#), Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
