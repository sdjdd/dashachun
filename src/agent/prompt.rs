use std::sync::LazyLock;

use minijinja::{Environment, Value, context, syntax::SyntaxConfig};
use serde::Serialize;

use crate::agent::ChatItem;
use crate::agent::memory::{MAX_ENTRIES, MemoryEntry, entry_id};

/// The whole system prompt in one template: the fixed voice and output
/// rules, then — each only when its data is present — the persona block,
/// the known-speakers roster (with the message-label legend) and the
/// long-term memory block. Rust only supplies the data; every layout and
/// conditional lives here.
const PROMPT_TEMPLATE: &str = r#"你是一位通过智能设备与用户交谈的语音助手，设备会用语音合成把你的回复朗读出来。用用户的语言回复，每次回复都保持简短、口语化。

## 情绪表达

可以用一个 emoji 表达情绪：把它放在回复的最开头，并且必须是任意情绪的黄色脸谱（如 😊 😄 😢 😠 🤔 😎）。正确：“😊 今天天气真好！” 错误：“👍 好的！”——任何非黄色脸谱的 emoji（点赞、爱心、动物、物品、星光等）在设备上没有任何效果，永远不要使用。

## 输出规则

- 只输出纯文本，不要使用 Markdown、代码块或列表符号。
- 不要在括号里写舞台指示、内心想法或动作描写。
{% if persona %}

## 角色设定

<persona>
{{ persona }}
</persona>
{% endif %}
{% if speakers %}

## 区分说话人

<known_speakers>
{% for speaker in speakers %}
  <speaker id="{{ speaker.id }}" name="{{ speaker.name }}">
{% if speaker.description %}
    <description>{{ speaker.description }}</description>
{% endif %}
  </speaker>
{% endfor %}
</known_speakers>

每条用户消息开头的 [speaker_id={id}] 标记了说话人：{id} 对应上述名册中的一位，unknown 表示未能识别。
任何未识别的消息都会被标记为 unknown，不应假设所有 unknown 消息都属于同一个人。
{% endif %}
{% if entries %}

## 长期记忆

你在过去的交谈中记住的事：

<memories>
{% for entry in entries %}
  <memory id="{{ entry.id }}">{{ entry.content }}</memory>
{% endfor %}
</memories>

你可以通过 memory_add 来记住一件事，保持简短、客观。
通过 memory_update 更正已经记住的事，通过 memory_delete 忘记已经记住的事。
你最多记住 {{ max_entries }} 件事。

{% endif %}"#;

static TEMPLATES: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_syntax(
        SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("valid syntax config"),
    );
    env.add_template("prompt", PROMPT_TEMPLATE)
        .expect("valid prompt template");
    env
});

fn render(ctx: Value) -> String {
    TEMPLATES
        .get_template("prompt")
        .expect("registered prompt template")
        .render(ctx)
        .expect("valid prompt template")
}

/// Renders the whole prompt from its data: the persona, the agent's known
/// speakers and the scope's memory entries. Absent data renders the
/// corresponding section away.
fn render_prompt(persona: &str, entries: &[MemoryEntry], speakers: &[Speaker]) -> String {
    let entries = entries
        .iter()
        .map(|entry| {
            context! {
                id => entry_id(entry.mem_no),
                content => &entry.content,
            }
        })
        .collect::<Value>();
    let speakers = speakers
        .iter()
        .map(|speaker| {
            context! {
                id => speaker.id,
                name => &speaker.name,
                description => &speaker.description,
            }
        })
        .collect::<Value>();
    render(context! {
        persona,
        speakers,
        entries,
        max_entries => MAX_ENTRIES,
    })
}

/// The user message the LLM context carries: the utterance's raw ASR text
/// prefixed with its speaker when the agent knows any — the matched
/// voiceprint row's id, the same number the roster renders, `unknown` for
/// an utterance that identified no one — and the raw text when the agent
/// has no roster at all. The prefix exists only in the warm context; the
/// stored transcript keeps the raw text.
pub fn labelled_utterance(text: String, speaker: Option<i64>, roster: &[Speaker]) -> String {
    if roster.is_empty() {
        return text;
    }
    let id = speaker.map_or_else(|| "unknown".to_string(), |id| id.to_string());
    format!("[speaker_id={id}] {text}")
}

/// One speaker the agent knows: an enrolled voiceprint — its row id as the
/// stable speaker id (the bare number in the roster and in the context's
/// message labels), its label as the name and its description. The
/// session's roster is snapshotted once per connection by the factory and
/// is turn-invariant.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Speaker {
    pub id: i64,
    pub name: String,
    pub description: String,
}

/// The voice assistant's system prompt, owned by the agent and injected into
/// every LLM request as the leading `ChatItem::System`.
///
/// Rendered whole from the one prompt template — the fixed voice and output
/// rules plus the optional persona block (from the device's bound agent in
/// the DB), the known-speakers roster and the scope's long-term memory
/// block. The `Default` value carries no prompt at all (no system message);
/// use [`Self::new`] for a real prompt.
#[derive(Clone, Debug, Default)]
pub struct SystemPrompt {
    persona: Option<String>,
    rendered: String,
}

impl SystemPrompt {
    pub fn new(persona: &str) -> Self {
        let persona = persona.trim();
        Self {
            persona: Some(persona.to_string()),
            rendered: render_prompt(persona, &[], &[]),
        }
    }

    /// The prompt for one turn: the whole template rendered with this
    /// turn's data — the scope's memory entries (reloaded every turn so
    /// they always reflect what the tools have mutated) and the agent's
    /// known speakers (turn-invariant: the per-utterance attribution rides
    /// the message labels, see [`labelled_utterance`]). Without entries and
    /// speakers the render is exactly the base prompt; `Default` (no prompt
    /// configured) stays empty.
    pub fn for_turn(&self, entries: &[MemoryEntry], speakers: &[Speaker]) -> Self {
        let Some(persona) = self.persona.as_deref() else {
            return Self::default();
        };
        Self {
            persona: Some(persona.to_string()),
            rendered: render_prompt(persona, entries, speakers),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.rendered
    }

    pub fn is_empty(&self) -> bool {
        self.rendered.is_empty()
    }

    pub fn chat_item(&self) -> Option<ChatItem> {
        if self.rendered.is_empty() {
            None
        } else {
            Some(ChatItem::system(self.rendered.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_prompt_embeds_the_persona_block() {
        let prompt = SystemPrompt::new("做一位有帮助的助手。");
        let content = prompt.as_str();
        assert!(content.starts_with("你是一位通过智能设备与用户交谈的语音助手"));
        assert!(content.contains("\n\n## 角色设定\n\n<persona>\n做一位有帮助的助手。\n</persona>"));
        assert!(content.find("输出规则").unwrap() < content.find("<persona>").unwrap());
        assert!(content.ends_with("</persona>\n"));
    }

    #[test]
    fn an_empty_persona_omits_the_block() {
        let prompt = SystemPrompt::new("  ");
        assert!(
            prompt
                .as_str()
                .starts_with("你是一位通过智能设备与用户交谈的语音助手")
        );
        assert!(prompt.as_str().ends_with("动作描写。\n"));
        assert!(prompt.chat_item().is_some());
    }

    #[test]
    fn the_emoji_rule_carries_its_examples() {
        let prompt = SystemPrompt::new("x");
        let content = prompt.as_str();
        assert!(content.contains("正确：“😊 今天天气真好！”"));
        assert!(content.contains("错误：“👍 好的！”"));
        assert!(content.contains("永远不要使用"));
    }

    #[test]
    fn default_is_empty_and_has_no_chat_item() {
        let prompt = SystemPrompt::default();
        assert!(prompt.is_empty());
        assert!(prompt.chat_item().is_none());
    }

    #[test]
    fn chat_item_carries_prompt() {
        let prompt = SystemPrompt::new("做一位有帮助的助手。");
        assert_eq!(prompt.chat_item(), Some(ChatItem::system(prompt.as_str())));
    }

    fn entry(mem_no: i32, content: &str) -> MemoryEntry {
        MemoryEntry {
            mem_no,
            content: content.to_string(),
        }
    }

    #[test]
    fn the_memory_section_lists_entries_and_guidance() {
        let prompt = SystemPrompt::new("认真回答。");
        let turn = prompt.for_turn(&[entry(1, "用 Rust 写后端"), entry(12, "喜欢喝茶")], &[]);
        let turn = turn.as_str();
        assert!(turn.contains(
            "\n\n<memories>\n  \
             <memory id=\"mem_01\">用 Rust 写后端</memory>\n  \
             <memory id=\"mem_12\">喜欢喝茶</memory>\n\
             </memories>\n\n"
        ));
        assert!(turn.contains("memory_add"));
        assert!(turn.contains("memory_update"));
        assert!(turn.contains("memory_delete"));
        assert!(turn.contains(&format!("你最多记住 {MAX_ENTRIES} 件事")));
    }

    #[test]
    fn without_entries_or_speakers_the_turn_prompt_is_the_base_untouched() {
        let prompt = SystemPrompt::new("认真回答。");
        assert_eq!(prompt.for_turn(&[], &[]).as_str(), prompt.as_str());
        assert!(SystemPrompt::default().for_turn(&[], &[]).is_empty());
    }

    #[test]
    fn for_turn_appends_the_memory_block_to_the_base() {
        let prompt = SystemPrompt::new("认真回答。");
        let turn = prompt.for_turn(&[entry(3, "喜欢喝茶")], &[]);
        let turn = turn.as_str();
        assert!(turn.starts_with(prompt.as_str()));
        assert!(turn.contains("\n\n## 长期记忆"));
        assert!(turn.contains("<memory id=\"mem_03\">喜欢喝茶</memory>"));
    }

    #[test]
    fn a_persona_less_base_still_carries_the_rules_and_the_memory() {
        let prompt = SystemPrompt::new("  ");
        let turn = prompt.for_turn(&[entry(1, "喜欢喝茶")], &[]);
        let turn = turn.as_str();
        assert!(turn.starts_with("你是一位通过智能设备与用户交谈的语音助手"));
        assert!(!turn.contains("<persona>"));
        assert!(turn.contains("<memory id=\"mem_01\">喜欢喝茶</memory>"));
    }

    #[test]
    fn the_default_prompt_stays_empty_even_with_sections() {
        let speakers = vec![speaker(1, "张三", "同事")];
        assert!(
            SystemPrompt::default()
                .for_turn(&[entry(1, "喜欢喝茶")], &speakers)
                .is_empty()
        );
        assert!(SystemPrompt::default().for_turn(&[], &speakers).is_empty());
    }

    fn speaker(id: i64, name: &str, description: &str) -> Speaker {
        Speaker {
            id,
            name: name.into(),
            description: description.into(),
        }
    }

    #[test]
    fn the_known_speakers_block_lists_the_roster_and_the_label_legend() {
        let prompt = SystemPrompt::new("认真回答。");
        let speakers = vec![
            speaker(1, "张三", "同事，喜欢喝茶"),
            speaker(12, "小红", ""),
        ];
        let turn = prompt.for_turn(&[], &speakers).as_str().to_string();
        assert!(turn.starts_with(prompt.as_str()));
        assert!(turn.contains(
            "\n\n## 区分说话人\n\n<known_speakers>\n  \
             <speaker id=\"1\" name=\"张三\">\n    \
             <description>同事，喜欢喝茶</description>\n  </speaker>\n  \
             <speaker id=\"12\" name=\"小红\">\n  </speaker>\n\
             </known_speakers>\n\n\
             每条用户消息开头的 [speaker_id={id}] 标记了说话人：{id} 对应上述名册中的一位，unknown 表示未能识别。\n\
             任何未识别的消息都会被标记为 unknown，不应假设所有 unknown 消息都属于同一个人。\n"
        ));
        assert!(!turn.contains("current="));
        assert!(!turn.contains("## 长期记忆"));
    }

    #[test]
    fn a_multiline_description_renders_verbatim() {
        let prompt = SystemPrompt::new("认真回答。");
        let description = "一位喜欢编程的用户。\n目前正在学习 Rust。";
        let turn = prompt
            .for_turn(&[], &[speaker(7, "小明", description)])
            .as_str()
            .to_string();
        assert!(turn.contains(&format!(
            "<speaker id=\"7\" name=\"小明\">\n    \
             <description>{description}</description>\n  </speaker>"
        )));
    }

    #[test]
    fn an_empty_roster_renders_no_block() {
        let prompt = SystemPrompt::new("认真回答。");
        let turn = prompt.for_turn(&[], &[]).as_str().to_string();
        assert!(!turn.contains("known_speakers"));
    }

    #[test]
    fn for_turn_orders_the_speakers_before_the_memory() {
        let prompt = SystemPrompt::new("认真回答。");
        let turn = prompt
            .for_turn(&[entry(3, "喜欢喝茶")], &[speaker(1, "张三", "")])
            .as_str()
            .to_string();
        let speakers = turn.find("<known_speakers>").unwrap();
        let memory = turn.find("## 长期记忆").unwrap();
        assert!(speakers < memory);
        assert!(turn.contains("同一个人。\n\n## 长期记忆"));
        assert!(turn.contains("<memory id=\"mem_03\">喜欢喝茶</memory>"));
    }

    #[test]
    fn a_rostered_agent_labels_its_utterances_with_the_row_id() {
        let roster = vec![speaker(1, "张三", "同事")];
        assert_eq!(
            labelled_utterance("你好".into(), Some(1), &roster),
            "[speaker_id=1] 你好"
        );
        assert_eq!(
            labelled_utterance("你好".into(), None, &roster),
            "[speaker_id=unknown] 你好"
        );
    }

    #[test]
    fn an_agent_without_a_roster_keeps_utterances_raw() {
        assert_eq!(labelled_utterance("你好".into(), Some(1), &[]), "你好");
        assert_eq!(labelled_utterance("你好".into(), None, &[]), "你好");
    }
}
