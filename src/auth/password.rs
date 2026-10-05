use argon2::password_hash::{PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng};
use argon2::{Algorithm, Argon2, Params, Version};

use crate::error::AppError;

const ARGON2_MEMORY_KIB: u32 = 19456;
const ARGON2_ITERATIONS: u32 = 2;
const ARGON2_PARALLELISM: u32 = 1;

fn hasher() -> Argon2<'static> {
    let params = Params::new(
        ARGON2_MEMORY_KIB,
        ARGON2_ITERATIONS,
        ARGON2_PARALLELISM,
        None,
    )
    .expect("valid argon2 params");
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
}

pub fn hash(password: &str) -> Result<String, AppError> {
    let salt = SaltString::generate(&mut OsRng);
    hasher()
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|err| AppError::Internal(format!("password hash failed: {err}")))
}

pub fn verify(password: &str, hash: &str) -> bool {
    let parsed = match argon2::PasswordHash::new(hash) {
        Ok(parsed) => parsed,
        Err(_) => return false,
    };
    hasher()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

pub fn verify_dummy(password: &str) {
    let _ = hash(password).map(|_| ());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_then_verify() {
        let hash = hash("correct horse battery staple").unwrap();
        assert!(verify("correct horse battery staple", &hash));
        assert!(!verify("wrong password", &hash));
    }

    #[test]
    fn hash_is_salted() {
        let a = hash("same password").unwrap();
        let b = hash("same password").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn verify_rejects_garbage_hash() {
        assert!(!verify("password", "not-a-phc-string"));
    }
}
