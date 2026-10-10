//! The real-storage suite: exercises [`Oss`] against the dev RustFS from
//! `compose.yaml` (`docker compose up -d rustfs`). Run with
//! `cargo test --test oss -- --ignored`.

use dashachun::agent::AudioStorage;
use dashachun::config::OssConfig;
use dashachun::oss::Oss;

fn oss() -> Oss {
    let config = OssConfig {
        endpoint: std::env::var("S3_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".into()),
        region: std::env::var("S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: std::env::var("S3_ACCESS_KEY_ID").unwrap_or_else(|_| "rustfsadmin".into()),
        secret_access_key: std::env::var("S3_SECRET_ACCESS_KEY")
            .unwrap_or_else(|_| "rustfsadmin".into()),
        force_path_style: true,
    };
    Oss::new(
        &config,
        &std::env::var("S3_BUCKET").unwrap_or_else(|_| "dsc-oss-test".into()),
    )
}

fn test_key() -> String {
    format!("oss-test/{}.bin", uuid::Uuid::new_v4().simple())
}

#[tokio::test]
#[ignore]
async fn ensure_bucket_is_idempotent() {
    let oss = oss();
    oss.ensure_bucket().await.unwrap();
    oss.ensure_bucket().await.unwrap();
}

#[tokio::test]
#[ignore]
async fn put_roundtrips_through_get() {
    let oss = oss();
    oss.ensure_bucket().await.unwrap();
    let key = test_key();
    let bytes: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
    oss.put(&key, bytes.clone()).await.unwrap();
    assert_eq!(oss.get(&key).await.unwrap(), bytes);
}

#[tokio::test]
#[ignore]
async fn get_of_a_missing_object_fails() {
    let oss = oss();
    oss.ensure_bucket().await.unwrap();
    assert!(oss.get(&test_key()).await.is_err());
}
