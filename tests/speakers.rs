//! The speakers suite: store CRUD, ownership scoping, and the
//! element-wise append semantics, plus the REST surface over the same
//! store. Run with `cargo test --test speakers -- --ignored`.

mod common;

use std::sync::Arc;

use axum::http::StatusCode;
use sqlx::PgPool;

use common::{
    MemoryStorage, app, app_with_state, app_with_voiceprint, conversation, expect_json, register,
    request, send,
};

use dashachun::agent::{AgentStore, AudioStorage};
use dashachun::error::AppError;
use dashachun::voiceprint::{DEFAULT_MODEL_PATH, SpeakerStore, Voiceprint};

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

async fn seed(pool: &PgPool, agent_id: i64, user_id: i64, name: &str) -> i64 {
    SpeakerStore::new(pool.clone())
        .create_or_append(agent_id, user_id, name, None, vec192(0.5), 1)
        .await
        .unwrap()
        .id
}

/// Encodes 16-bit PCM as FLAC, mirroring the capture path's tail padding.
fn flac_bytes(sample_rate: u32, channels: usize, samples: &[i16]) -> Vec<u8> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let config = flacenc::config::Encoder::default().into_verified().unwrap();
    let mut padded: Vec<i32> = samples.iter().map(|&s| i32::from(s)).collect();
    let remainder = padded.len() % config.block_size;
    if remainder != 0 && remainder < 16 {
        padded.extend(std::iter::repeat_n(0, 16 - remainder));
    }
    let source =
        flacenc::source::MemSource::from_samples(&padded, channels, 16, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size).unwrap();
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).unwrap();
    sink.as_slice().to_vec()
}

/// A synthetic utterance: summed sine waves at the given frequencies, one
/// second per `seconds` unit.
fn tone(freqs: &[f32], seconds: usize) -> Vec<i16> {
    let rate = 16000.0f32;
    (0..seconds * rate as usize)
        .map(|i| {
            let t = i as f32 / rate;
            let mixed: f32 = freqs
                .iter()
                .map(|f| (t * f * std::f32::consts::TAU).sin())
                .sum();
            (mixed / freqs.len() as f32 * 12000.0) as i16
        })
        .collect()
}

async fn message_row(pool: &PgPool, agent_id: i64, role: &str) -> i64 {
    let conversation_id = conversation(pool, 1, agent_id).await;
    let message = serde_json::json!({"role": role, "content": "你好"});
    sqlx::query_scalar(
        "INSERT INTO messages (conversation_id, message) VALUES ($1, $2) RETURNING id",
    )
    .bind(conversation_id)
    .bind(message)
    .fetch_one(pool)
    .await
    .unwrap()
}

/// Inserts a user message row plus its capture row, and puts the FLAC
/// object into the test storage under the recorded key.
async fn captured_message(
    pool: &PgPool,
    agent_id: i64,
    storage: &MemoryStorage,
    sample_rate: u32,
    channels: usize,
    samples: &[i16],
) -> i64 {
    let id = message_row(pool, agent_id, "user").await;
    let key = format!("audios/test/{}.flac", uuid::Uuid::new_v4().simple());
    storage
        .put(&key, flac_bytes(sample_rate, channels, samples))
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO message_audios \
         (message_id, bucket, object_key, sample_rate, channels, duration_ms) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(id)
    .bind(storage.bucket())
    .bind(&key)
    .bind(sample_rate as i32)
    .bind(channels as i32)
    .bind(((samples.len() / channels) * 1000 / sample_rate as usize) as i32)
    .execute(pool)
    .await
    .unwrap();
    id
}

async fn audio_row(pool: &PgPool, message_id: i64, bucket: &str, key: &str) {
    sqlx::query(
        "INSERT INTO message_audios \
         (message_id, bucket, object_key, sample_rate, channels, duration_ms) \
         VALUES ($1, $2, $3, 16000, 1, 1000)",
    )
    .bind(message_id)
    .bind(bucket)
    .bind(key)
    .execute(pool)
    .await
    .unwrap();
}

#[sqlx::test]
#[ignore]
async fn append_sums_elementwise_and_accumulates_sample_count(pool: PgPool) {
    let store = SpeakerStore::new(pool.clone());
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

    let embedding: Vec<f32> = sqlx::query_scalar("SELECT embedding FROM speakers WHERE id = $1")
        .bind(first.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(embedding.iter().all(|&x| x == 0.75));
}

#[sqlx::test]
#[ignore]
async fn create_or_append_on_an_agent_you_do_not_own_is_not_found(pool: PgPool) {
    let store = SpeakerStore::new(pool.clone());
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
async fn same_name_coexists_across_agents_and_lists_are_scoped(pool: PgPool) {
    let store = SpeakerStore::new(pool.clone());
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
async fn update_renames_and_conflicts_on_a_taken_name(pool: PgPool) {
    let store = SpeakerStore::new(pool.clone());
    let agent = owned_agent(&pool, 1, "a").await;
    let first = store
        .create_or_append(agent, 1, "a-name", None, vec192(0.5), 1)
        .await
        .unwrap();
    store
        .create_or_append(agent, 1, "b-name", None, vec192(0.5), 1)
        .await
        .unwrap();

    let renamed = store
        .update(agent, first.id, 1, Some("c-name"), Some("新描述"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(renamed.name, "c-name");
    assert_eq!(renamed.description, "新描述");

    let err = store
        .update(agent, first.id, 1, Some("b-name"), None)
        .await
        .unwrap_err();
    assert!(matches!(err, AppError::Conflict(_)));

    let kept = store
        .update(agent, first.id, 1, None, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(kept.name, "c-name");
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
    let store = SpeakerStore::new(pool.clone());
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
            &format!("/api/agents/{agent}/speakers"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body.as_array().unwrap().len(), 1);
    let entry = &body[0];
    assert_eq!(entry["name"], "张三");
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
            &format!("/api/agents/{agent}/speakers"),
            None,
            Some(&other_cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request("GET", "/api/agents/999999/speakers", None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request("GET", &format!("/api/agents/{agent}/speakers"), None, None),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn create_rejects_invalid_payloads_and_reports_disabled_audio_capture(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let path = format!("/api/agents/{agent}/speakers");

    for payload in [
        r#"{}"#,
        r#"{"name":"","message_ids":[1]}"#,
        r#"{"name":"张三","message_ids":[]}"#,
        r#"{"name":"张三","message_ids":[1,2,3,4,5,6,7,8,9,10,11]}"#,
    ] {
        let res = send(&app, request("POST", &path, Some(payload), Some(&cookie))).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "payload: {payload}");
    }

    let body = expect_json(
        &app,
        request(
            "POST",
            &path,
            Some(r#"{"name":"张三","message_ids":[1]}"#),
            Some(&cookie),
        ),
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("audio capture is not enabled"),
        "{}",
        body["error"]
    );

    let (other_cookie, other_user) = register(&app, &pool).await;
    let foreign_agent = owned_agent(&pool, other_user, "b").await;
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{foreign_agent}/speakers"),
            Some(r#"{"name":"x","message_ids":[1]}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent}/speakers"),
            Some(r#"{"name":"x","message_ids":[1]}"#),
            Some(&other_cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore]
async fn create_400s_on_out_of_scope_and_uncaptured_messages(pool: PgPool) {
    let (app, state) = app_with_state(pool.clone());
    state.set_audio_storage(Some(Arc::new(MemoryStorage::new("bucket"))));

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "mine").await;
    let other_agent = owned_agent(&pool, user_id, "other-mine").await;
    let (other_cookie, other_user) = register(&app, &pool).await;
    let foreign_agent = owned_agent(&pool, other_user, "theirs").await;

    let uncaptured = message_row(&pool, agent, "user").await;
    let assistant = message_row(&pool, agent, "assistant").await;
    let cross_agent = message_row(&pool, other_agent, "user").await;
    let foreign = message_row(&pool, foreign_agent, "user").await;

    let body = expect_json(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent}/speakers"),
            Some(&format!(
                r#"{{"name":"张三","message_ids":[{uncaptured},{assistant},{cross_agent},{foreign},999999]}}"#
            )),
            Some(&cookie),
        ),
        StatusCode::BAD_REQUEST,
    )
    .await;
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("no audio captured"), "{error}");
    assert!(error.contains("not a user message"), "{error}");
    assert_eq!(error.matches("not in scope").count(), 3, "{error}");

    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent}/speakers"),
            Some(r#"{"name":"x","message_ids":[1]}"#),
            Some(&other_cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore]
async fn create_400s_on_missing_objects_and_undecodable_audio(pool: PgPool) {
    let (app, state) = app_with_state(pool.clone());
    let storage = Arc::new(MemoryStorage::new("bucket"));
    state.set_audio_storage(Some(storage.clone()));

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let path = format!("/api/agents/{agent}/speakers");

    let missing = message_row(&pool, agent, "user").await;
    audio_row(&pool, missing, "bucket", "never-stored.flac").await;
    let broken = message_row(&pool, agent, "user").await;
    audio_row(&pool, broken, "bucket", "broken.flac").await;
    storage.put("broken.flac", vec![0u8; 64]).await.unwrap();

    let body = expect_json(
        &app,
        request(
            "POST",
            &path,
            Some(&format!(
                r#"{{"name":"张三","message_ids":[{missing},{broken}]}}"#
            )),
            Some(&cookie),
        ),
        StatusCode::BAD_REQUEST,
    )
    .await;
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("audio storage"), "{error}");
    assert!(error.contains("audio decode failed"), "{error}");
}

#[sqlx::test]
#[ignore]
async fn create_503s_when_the_audio_is_valid_but_the_model_is_absent(pool: PgPool) {
    let (app, state) = app_with_state(pool.clone());
    let storage = Arc::new(MemoryStorage::new("bucket"));
    state.set_audio_storage(Some(storage.clone()));

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let id = captured_message(
        &pool,
        agent,
        &storage,
        16000,
        1,
        &tone(&[220.0, 440.0, 660.0], 1),
    )
    .await;

    let res = send(
        &app,
        request(
            "POST",
            &format!("/api/agents/{agent}/speakers"),
            Some(&format!(r#"{{"name":"张三","message_ids":[{id}]}}"#)),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[sqlx::test]
#[ignore]
async fn create_enrolls_and_appends_over_rest(pool: PgPool) {
    let model_path =
        std::env::var("VOICEPRINT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    let model = Arc::new(tokio::sync::Mutex::new(
        Voiceprint::load(&model_path).unwrap(),
    ));
    let (app, state) = app_with_voiceprint(pool.clone(), model.clone());
    let storage = Arc::new(MemoryStorage::new("bucket"));
    state.set_audio_storage(Some(storage.clone()));

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let path = format!("/api/agents/{agent}/speakers");

    let first = tone(&[220.0, 440.0, 660.0], 1);
    let second = tone(&[300.0, 900.0], 1);
    let m1 = captured_message(&pool, agent, &storage, 16000, 1, &first).await;
    let m2 = captured_message(&pool, agent, &storage, 16000, 1, &second).await;

    let body = expect_json(
        &app,
        request(
            "POST",
            &path,
            Some(&format!(
                r#"{{"name":"张三","description":"主人","message_ids":[{m1}]}}"#
            )),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(body["sample_count"], 1);
    assert_eq!(body["description"], "主人");
    let vp_id = body["id"].as_i64().unwrap();

    let expected: Vec<f32> = {
        let mut guard = model.lock().await;
        let embedding = guard.embed(&first).unwrap();
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        embedding.iter().map(|v| v / norm).collect()
    };
    let stored: Vec<f32> = sqlx::query_scalar("SELECT embedding FROM speakers WHERE id = $1")
        .bind(vp_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(stored.len(), expected.len());
    for (a, b) in stored.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    let body = expect_json(
        &app,
        request(
            "POST",
            &path,
            Some(&format!(r#"{{"name":"李四","message_ids":[{m1}]}}"#)),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    let same_audio: Vec<f32> = sqlx::query_scalar("SELECT embedding FROM speakers WHERE id = $1")
        .bind(body["id"].as_i64().unwrap())
        .fetch_one(&pool)
        .await
        .unwrap();
    let dot: f32 = stored.iter().zip(&same_audio).map(|(a, b)| a * b).sum();
    assert!(dot > 0.999, "same audio must score identical: {dot}");

    let body = expect_json(
        &app,
        request(
            "POST",
            &path,
            Some(&format!(r#"{{"name":"张三","message_ids":[{m2}]}}"#)),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(body["id"], vp_id);
    assert_eq!(body["sample_count"], 2);

    let expected2 = {
        let mut guard = model.lock().await;
        let embedding = guard.embed(&second).unwrap();
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut sum = expected.clone();
        for (acc, value) in sum.iter_mut().zip(&embedding) {
            *acc += value / norm;
        }
        sum
    };
    let appended: Vec<f32> = sqlx::query_scalar("SELECT embedding FROM speakers WHERE id = $1")
        .bind(vp_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    for (a, b) in appended.iter().zip(&expected2) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}

#[sqlx::test]
#[ignore]
async fn embeddings_for_agent_reads_the_identification_vectors(pool: PgPool) {
    let store = SpeakerStore::new(pool.clone());
    let mine = owned_agent(&pool, 1, "mine").await;
    let theirs = owned_agent(&pool, 2, "theirs").await;
    store
        .create_or_append(mine, 1, "张三", Some("同事"), vec192(0.5), 1)
        .await
        .unwrap();
    store
        .create_or_append(theirs, 2, "李四", None, vec192(0.5), 1)
        .await
        .unwrap();

    // The identification read is agent-scoped and carries the description
    // and the vector the REST surface hides.
    let embeddings = store.embeddings_for_agent(mine).await.unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].name, "张三");
    assert_eq!(embeddings[0].description, "同事");
    assert_eq!(embeddings[0].embedding, vec192(0.5));

    assert!(
        store
            .embeddings_for_agent(999_999)
            .await
            .unwrap()
            .is_empty()
    );
}

#[sqlx::test]
#[ignore]
async fn update_and_delete_over_rest_are_scoped_by_ownership(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let first = seed(&pool, agent, user_id, "first").await;
    seed(&pool, agent, user_id, "second").await;
    let path = format!("/api/agents/{agent}/speakers/{first}");

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
    assert_eq!(body["name"], "first");
    assert_eq!(body["description"], "新的描述");

    for payload in [
        r#"{}"#,
        r#"{"name":""}"#,
        &format!(r#"{{"description":"{}"}}"#, "长".repeat(201)),
    ] {
        let res = send(&app, request("PUT", &path, Some(payload), Some(&cookie))).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "payload: {payload}");
    }

    let res = send(
        &app,
        request("PUT", &path, Some(r#"{"name":"second"}"#), Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CONFLICT);

    let (other_cookie, _) = register(&app, &pool).await;
    for (method, payload) in [("PUT", Some(r#"{"name":"stolen"}"#)), ("DELETE", None)] {
        let res = send(&app, request(method, &path, payload, Some(&other_cookie))).await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND, "{method}");
    }

    let res = send(&app, request("DELETE", &path, None, Some(&cookie))).await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    let res = send(&app, request("DELETE", &path, None, Some(&cookie))).await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request("PUT", &path, Some(r#"{"name":"third"}"#), Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
