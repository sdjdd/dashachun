//! The S3-compatible object storage behind the audio capture: a thin
//! [`aws_sdk_s3`] wrapper built directly from [`OssConfig`] (no env-based
//! credential resolution — the boot config is the only source), implementing
//! the capture feature's [`AudioStorage`] port.

use aws_sdk_s3::Client;
use aws_sdk_s3::config::{BehaviorVersion, Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;

use crate::agent::AudioStorage;
use crate::config::OssConfig;

pub struct Oss {
    client: Client,
    bucket: String,
}

impl Oss {
    pub fn new(storage: &OssConfig, bucket: &str) -> Self {
        let s3 = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(storage.region.clone()))
            .endpoint_url(storage.endpoint.clone())
            .force_path_style(storage.force_path_style)
            .credentials_provider(Credentials::new(
                storage.access_key_id.clone(),
                storage.secret_access_key.clone(),
                None,
                None,
                "OssConfig",
            ))
            .build();
        Self {
            client: Client::from_conf(s3),
            bucket: bucket.to_owned(),
        }
    }

    /// Creates the bucket when missing. Idempotent: an already-owned bucket
    /// answers the create with `BucketAlreadyOwnedByYou`.
    pub async fn ensure_bucket(&self) -> Result<(), String> {
        match self
            .client
            .create_bucket()
            .bucket(&self.bucket)
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(err) => {
                let err = err.into_service_error();
                if err.is_bucket_already_owned_by_you() || err.is_bucket_already_exists() {
                    Ok(())
                } else {
                    Err(format!("ensure bucket {}: {err}", self.bucket))
                }
            }
        }
    }

    /// Reads one object back. Not on the capture path — diagnostics and the
    /// real-storage test suite verify roundtrips through it.
    pub async fn get(&self, key: &str) -> Result<Vec<u8>, String> {
        let object = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|err| format!("get {key}: {err}"))?;
        let data = object
            .body
            .collect()
            .await
            .map_err(|err| format!("read {key}: {err}"))?;
        Ok(data.into_bytes().to_vec())
    }
}

#[async_trait::async_trait]
impl AudioStorage for Oss {
    /// Writes one object and returns the bucket it landed in — the row the
    /// capture inserts must record it, so a later bucket switch can never
    /// orphan the reference.
    async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<String, String> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes))
            .send()
            .await
            .map_err(|err| format!("put {key}: {err}"))?;
        Ok(self.bucket.clone())
    }
}
