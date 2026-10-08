use crate::agent::ChatItem;

/// Fixed prefix prepended to every system message, before the configurable
/// persona prompt. Carries the TTS plain-text output rules and the
/// single-leading-emoji directive the emotion pipeline relies on.
pub const SYSTEM_PROMPT_PREFIX: &str = "\
You are a voice assistant speaking through a device that reads your replies \
aloud with text-to-speech. Reply in the user's language and keep every reply \
short and conversational.

TTS output rules:
- Output plain text only. Never use Markdown, code blocks, or bullet points.
- Never write stage directions, inner thoughts, or actions in brackets or \
parentheses.
- To express emotion, start a regular reply with exactly one emoji. Never \
place the emoji anywhere else.
- When calling a tool, output only the tool call with no emoji and no text.";

/// The voice assistant's system prompt, owned by the agent and injected into
/// every LLM request as the leading `ChatItem::System`.
///
/// Built from the fixed [`SYSTEM_PROMPT_PREFIX`] plus an optional persona
/// block (the persona comes from the device's bound agent in the DB). The
/// `Default` value is empty (no system message); use [`Self::new`] for a real
/// prompt.
#[derive(Clone, Debug, Default)]
pub struct SystemPrompt(String);

impl SystemPrompt {
    pub fn new(persona: &str) -> Self {
        let persona = persona.trim();
        if persona.is_empty() {
            return Self(SYSTEM_PROMPT_PREFIX.to_string());
        }
        Self(format!(
            "{SYSTEM_PROMPT_PREFIX}\n\n<persona>\n{persona}\n</persona>"
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn chat_item(&self) -> Option<ChatItem> {
        if self.0.is_empty() {
            None
        } else {
            Some(ChatItem::system(self.0.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_precedes_persona_block() {
        let prompt = SystemPrompt::new("Be a helpful assistant.");
        let content = prompt.as_str();
        assert!(content.starts_with(SYSTEM_PROMPT_PREFIX));
        assert!(content.contains("<persona>\nBe a helpful assistant.\n</persona>"));
        assert!(content.find("TTS output rules").unwrap() < content.find("<persona>").unwrap());
    }

    #[test]
    fn empty_persona_omits_block() {
        let prompt = SystemPrompt::new("");
        assert_eq!(prompt.as_str(), SYSTEM_PROMPT_PREFIX);
        assert!(!prompt.as_str().contains("<persona>"));
    }

    #[test]
    fn default_is_empty_and_has_no_chat_item() {
        let prompt = SystemPrompt::default();
        assert!(prompt.is_empty());
        assert!(prompt.chat_item().is_none());
    }

    #[test]
    fn chat_item_carries_prompt() {
        let prompt = SystemPrompt::new("Be a helpful assistant.");
        assert_eq!(prompt.chat_item(), Some(ChatItem::system(prompt.as_str())));
    }
}
