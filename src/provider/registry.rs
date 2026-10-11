use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use super::{asr, llm, tts};
use crate::agent::{Asr, Llm, Tts};

pub type AsrBuilder = fn(&Value) -> Result<Arc<dyn Asr>, String>;
pub type LlmBuilder = fn(&Value) -> Result<Arc<dyn Llm>, String>;
pub type TtsBuilder = fn(&Value) -> Result<Arc<dyn Tts>, String>;

#[derive(Clone)]
pub struct ProviderSpec {
    pub kind: String,
    pub config: Value,
}

impl ProviderSpec {
    pub fn new(kind: impl Into<String>, config: Value) -> Self {
        Self {
            kind: kind.into(),
            config,
        }
    }

    pub fn stub() -> Self {
        Self::new("stub", serde_json::json!({}))
    }
}

#[derive(Default)]
pub struct ProviderRegistry {
    asr: HashMap<&'static str, AsrBuilder>,
    llm: HashMap<&'static str, LlmBuilder>,
    tts: HashMap<&'static str, TtsBuilder>,
}

impl ProviderRegistry {
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        asr::volcengine::register(&mut registry);
        llm::openai::register(&mut registry);
        tts::volcengine::register(&mut registry);
        registry
    }

    pub fn with_stubs() -> Self {
        let mut registry = Self::default();
        asr::register_stub(&mut registry);
        llm::register_stub(&mut registry);
        tts::register_stub(&mut registry);
        registry
    }

    pub fn register_asr(&mut self, kind: &'static str, build: AsrBuilder) {
        if self.asr.insert(kind, build).is_some() {
            panic!("asr provider kind \"{kind}\" registered twice");
        }
    }

    pub fn register_llm(&mut self, kind: &'static str, build: LlmBuilder) {
        if self.llm.insert(kind, build).is_some() {
            panic!("llm provider kind \"{kind}\" registered twice");
        }
    }

    pub fn register_tts(&mut self, kind: &'static str, build: TtsBuilder) {
        if self.tts.insert(kind, build).is_some() {
            panic!("tts provider kind \"{kind}\" registered twice");
        }
    }

    pub fn build_asr(&self, kind: &str, config: &Value) -> Result<Arc<dyn Asr>, String> {
        lookup(&self.asr, "asr", kind)?(config)
    }

    pub fn build_llm(&self, kind: &str, config: &Value) -> Result<Arc<dyn Llm>, String> {
        lookup(&self.llm, "llm", kind)?(config)
    }

    pub fn build_tts(&self, kind: &str, config: &Value) -> Result<Arc<dyn Tts>, String> {
        lookup(&self.tts, "tts", kind)?(config)
    }

    /// Checks a type/kind/config triple against the registered builders by
    /// constructing the provider and dropping it — the create/update path
    /// validates eagerly so a stored config can never fail activation on
    /// schema errors.
    pub fn validate(&self, r#type: &str, kind: &str, config: &Value) -> Result<(), String> {
        match r#type {
            "asr" => self.build_asr(kind, config).map(|_| ()),
            "llm" => self.build_llm(kind, config).map(|_| ()),
            "tts" => self.build_tts(kind, config).map(|_| ()),
            other => Err(format!(
                "unknown provider type \"{other}\" (expected asr, llm or tts)"
            )),
        }
    }

    /// Resolves an omitted `kind` to the sole registered implementation of a
    /// type. The error names the registered kinds when the choice would be
    /// ambiguous — the contract for when a type grows a second implementation.
    pub fn sole_kind(&self, r#type: &str) -> Result<&'static str, String> {
        match self.kinds_of(r#type)?.as_slice() {
            [only] => Ok(only),
            [] => Err(format!(
                "no provider kind is registered for type \"{type}\""
            )),
            many => Err(format!(
                "several provider kinds are registered for type \"{type}\" ({many:?}); pass \"kind\" explicitly"
            )),
        }
    }

    fn kinds_of(&self, r#type: &str) -> Result<Vec<&'static str>, String> {
        match r#type {
            "asr" => Ok(self.asr_kinds()),
            "llm" => Ok(self.llm_kinds()),
            "tts" => Ok(self.tts_kinds()),
            other => Err(format!(
                "unknown provider type \"{other}\" (expected asr, llm or tts)"
            )),
        }
    }

    pub fn asr_kinds(&self) -> Vec<&'static str> {
        kinds(&self.asr)
    }

    pub fn llm_kinds(&self) -> Vec<&'static str> {
        kinds(&self.llm)
    }

    pub fn tts_kinds(&self) -> Vec<&'static str> {
        kinds(&self.tts)
    }
}

fn lookup<'m, B>(
    stage: &'m HashMap<&'static str, B>,
    stage_name: &str,
    kind: &str,
) -> Result<&'m B, String> {
    stage.get(kind).ok_or_else(|| {
        format!(
            "unknown {stage_name} provider kind \"{kind}\" (registered: {:?})",
            kinds(stage)
        )
    })
}

fn kinds<B>(stage: &HashMap<&'static str, B>) -> Vec<&'static str> {
    let mut kinds: Vec<&'static str> = stage.keys().copied().collect();
    kinds.sort_unstable();
    kinds
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn err_of<T>(result: Result<T, String>) -> String {
        result.err().expect("expected a build error")
    }

    #[test]
    fn builtins_register_one_kind_per_stage() {
        let registry = ProviderRegistry::with_builtins();
        assert_eq!(registry.asr_kinds(), vec!["volcengine"]);
        assert_eq!(registry.llm_kinds(), vec!["openai"]);
        assert_eq!(registry.tts_kinds(), vec!["volcengine"]);
    }

    #[test]
    fn builds_the_volc_asr_from_its_config() {
        let registry = ProviderRegistry::with_builtins();
        registry
            .build_asr(
                "volcengine",
                &json!({"base_url": "wss://asr", "api_key": "k", "resource_id": "r"}),
            )
            .unwrap();
    }

    #[test]
    fn builds_the_openai_llm_with_optional_fields() {
        let registry = ProviderRegistry::with_builtins();
        registry
            .build_llm(
                "openai",
                &json!({"base_url": "http://localhost/v1", "api_key": "k", "model": "m"}),
            )
            .unwrap();
    }

    #[tokio::test]
    async fn builds_the_volc_tts_from_its_config() {
        let registry = ProviderRegistry::with_builtins();
        registry
            .build_tts(
                "volcengine",
                &json!({
                    "base_url": "wss://tts",
                    "api_key": "k",
                    "resource_id": "r",
                    "speaker": "s"
                }),
            )
            .unwrap();
    }

    #[test]
    fn missing_required_config_fields_fail_the_build() {
        let registry = ProviderRegistry::with_builtins();
        let err = err_of(registry.build_asr(
            "volcengine",
            &json!({"base_url": "wss://asr", "api_key": "k"}),
        ));
        assert!(err.contains("resource_id"), "unexpected error: {err}");
        let err = err_of(registry.build_tts(
            "volcengine",
            &json!({"base_url": "wss://tts", "api_key": "k", "resource_id": "r"}),
        ));
        assert!(err.contains("speaker"), "unexpected error: {err}");
        let err = err_of(registry.build_llm("openai", &json!({"base_url": "http://localhost/v1"})));
        assert!(err.contains("api_key"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_kinds_name_the_registered_kinds() {
        let registry = ProviderRegistry::with_builtins();
        let err = err_of(registry.build_asr("aliyun", &json!({})));
        assert!(
            err.contains("unknown asr provider kind") && err.contains("volcengine"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_passes_registered_stage_kind_config_triples() {
        let registry = ProviderRegistry::with_builtins();
        registry
            .validate(
                "asr",
                "volcengine",
                &json!({"base_url": "wss://asr", "api_key": "k", "resource_id": "r"}),
            )
            .unwrap();
        registry
            .validate(
                "llm",
                "openai",
                &json!({"base_url": "http://localhost/v1", "api_key": "k", "model": "m"}),
            )
            .unwrap();
    }

    #[test]
    fn validate_rejects_unknown_types_and_broken_configs() {
        let registry = ProviderRegistry::with_builtins();
        let err = err_of(registry.validate("mic", "volcengine", &json!({})));
        assert!(
            err.contains("unknown provider type") && err.contains("mic"),
            "unexpected error: {err}"
        );
        let err = err_of(registry.validate("asr", "volcengine", &json!({"base_url": "wss://asr"})));
        assert!(err.contains("api_key"), "unexpected error: {err}");
    }

    #[test]
    fn sole_kind_resolves_the_only_registered_implementation() {
        let registry = ProviderRegistry::with_builtins();
        assert_eq!(registry.sole_kind("asr").unwrap(), "volcengine");
        assert_eq!(registry.sole_kind("llm").unwrap(), "openai");
        assert_eq!(registry.sole_kind("tts").unwrap(), "volcengine");
    }

    #[test]
    fn sole_kind_rejects_unknown_types_and_ambiguous_choices() {
        let registry = ProviderRegistry::with_builtins();
        let err = err_of(registry.sole_kind("mic"));
        assert!(
            err.contains("unknown provider type") && err.contains("mic"),
            "unexpected error: {err}"
        );

        let mut ambiguous = ProviderRegistry::with_builtins();
        crate::provider::asr::register_stub(&mut ambiguous);
        let err = err_of(ambiguous.sole_kind("asr"));
        assert!(
            err.contains("several provider kinds") && err.contains("pass \"kind\" explicitly"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[should_panic(expected = "registered twice")]
    fn duplicate_kind_registration_panics() {
        fn builder(_: &Value) -> Result<Arc<dyn Asr>, String> {
            unimplemented!()
        }
        let mut registry = ProviderRegistry::default();
        registry.register_asr("volcengine", builder);
        registry.register_asr("volcengine", builder);
    }
}
