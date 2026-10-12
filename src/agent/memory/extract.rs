//! The memory extractor: distills conversation batches into entry-memory
//! changes in the background. One extractor per connection implements
//! [`MemoryHook`] and owns exactly the rows its own connection wrote — the
//! announced ids buffer in memory until [`BATCH_ROWS`] of them are pending,
//! then one flush reads them back from the log and puts a forced
//! [`MemoryPropose`] decision through the generation loop; dropping the
//! connection flushes whatever is left. There is no durable extraction
//! state: rows loaded at connection time count as seen (a past connection
//! either flushed them or died), so a crash only ever loses the crashed
//! connection's unflushed tail.

use std::sync::{Arc, LazyLock, Mutex};

use minijinja::{Environment, Value, context, syntax::SyntaxConfig};
use sqlx::PgPool;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
use tracing::{debug, warn};

use crate::agent::tool::{MemoryPropose, ToolEntry, stop_after_proposal};
use crate::agent::{
    AgentSession, ChatItem, Llm, ReasoningEffort, StreamTextParams, Tool, ToolChoice, generate_text,
};

use super::{EntryMemory, MAX_ENTRIES, MemoryEntry, MemoryHook, MemoryOwner, entry_id};

/// Flushes once this many of the connection's own rows are pending — about
/// six turns of dialogue per extraction call.
const BATCH_ROWS: usize = 12;

const EXTRACT_SYSTEM: &str = r#"你是长期记忆提炼器。阅读一段你（助手）与用户的对话，对照现有记忆条目，把值得长期保留的信息整理成对条目的修改，然后通过 memory_propose 工具一次性提交：
- 只记稳定有用的事实：身份、偏好、习惯、重要关系、长期计划等；寒暄和一次性事务不记。
- 与现有条目重复或仅措辞不同的事实跳过；事实变化时更新（update）原条目，失效时删除（delete）。
- 条目接近上限（{{ max_entries }}）时优先更新或删除，而不是新增。
- 每条内容一句话，自包含，第三人称。
- 没有值得记录的就提交空列表。

## 现有记忆条目
{% if entries %}
{% for entry in entries %}
[{{ entry.id }}] {{ entry.content }}
{% endfor %}
{% else %}
（暂无）
{% endif %}"#;

const EXTRACT_USER: &str = r#"对话记录{% if days|length == 1 %}（{{ days|first }}）{% elif days|length > 1 %}（{{ days|first }} 至 {{ days|last }}）{% endif %}：

{% for line in dialogue %}
{{ line.speaker }}：{{ line.text }}
{% endfor %}"#;

static TEMPLATES: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_syntax(
        SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("valid syntax config"),
    );
    env.add_template("extract_system", EXTRACT_SYSTEM)
        .expect("valid extractor system template");
    env.add_template("extract_user", EXTRACT_USER)
        .expect("valid extractor user template");
    env
});

pub struct Extractor {
    pool: PgPool,
    owner: MemoryOwner,
    llm: Arc<dyn Llm>,
    entries: Arc<dyn EntryMemory>,
    propose: ToolEntry,
    state: Arc<Mutex<ExtractState>>,
    flush_lock: Arc<AsyncMutex<()>>,
}

#[derive(Default)]
struct ExtractState {
    pending: Vec<i64>,
}

impl Extractor {
    pub fn new(
        pool: PgPool,
        owner: MemoryOwner,
        llm: Arc<dyn Llm>,
        entries: Arc<dyn EntryMemory>,
    ) -> Self {
        let propose = ToolEntry::new(MemoryPropose::new(Arc::clone(&entries)));
        Self {
            pool,
            owner,
            llm,
            entries,
            propose,
            state: Arc::new(Mutex::new(ExtractState::default())),
            flush_lock: Arc::new(AsyncMutex::new(())),
        }
    }

    /// The state-carrying clone a detached flush task needs; the propose
    /// tool comes from the entry built once at construction.
    fn flush_task(&self) -> FlushTask {
        FlushTask {
            pool: self.pool.clone(),
            owner: self.owner.clone(),
            llm: Arc::clone(&self.llm),
            entries: Arc::clone(&self.entries),
            propose: self.propose.clone(),
            state: Arc::clone(&self.state),
        }
    }

    fn poke(&self, message_id: i64) {
        let mut state = self.state.lock().unwrap();
        state.pending.push(message_id);
        if state.pending.len() < BATCH_ROWS {
            return;
        }
        drop(state);
        self.spawn_flush();
    }

    /// Becomes the flusher unless one is already running — the try-lock is
    /// the in-flight gate — and hands the drained batch to a detached task,
    /// which keeps going while full batches remain.
    fn spawn_flush(&self) {
        let Ok(guard) = Arc::clone(&self.flush_lock).try_lock_owned() else {
            return;
        };
        let task = self.flush_task();
        let batch = task.take_pending();
        if batch.is_empty() {
            return;
        }
        tokio::spawn(task.flush_all(guard, batch));
    }
}

impl MemoryHook for Extractor {
    fn utterance_stored(&self, _session: &AgentSession, message_id: i64) {
        self.poke(message_id);
    }

    fn turn_stored(&self, _session: &AgentSession, message_id: i64) {
        self.poke(message_id);
    }
}

impl Drop for Extractor {
    fn drop(&mut self) {
        if self.state.lock().unwrap().pending.is_empty() {
            return;
        }
        let task = self.flush_task();
        let lock = Arc::clone(&self.flush_lock);
        tokio::spawn(async move {
            // A flush may still be running — wait it out, then take whatever
            // is left, threshold or not: this is the connection's last
            // chance.
            let _guard = lock.lock().await;
            let batch = task.take_pending();
            if batch.is_empty() {
                return;
            }
            if let Err(err) = run_flush(&task, &batch).await {
                warn!(%err, count = batch.len(), "memory extraction teardown flush failed");
            }
        });
    }
}

/// The flush's state-carrying clone: everything a detached task needs to run
/// batches and restore them on failure.
struct FlushTask {
    pool: PgPool,
    owner: MemoryOwner,
    llm: Arc<dyn Llm>,
    entries: Arc<dyn EntryMemory>,
    propose: ToolEntry,
    state: Arc<Mutex<ExtractState>>,
}

impl FlushTask {
    fn take_pending(&self) -> Vec<i64> {
        std::mem::take(&mut self.state.lock().unwrap().pending)
    }

    /// Runs flushes back to back while full batches remain. The held guard
    /// keeps every poke that lands mid-flush buffered instead of spawning a
    /// second flush; the same guard serializes the teardown flush behind a
    /// still-running one.
    async fn flush_all(self, _guard: OwnedMutexGuard<()>, mut batch: Vec<i64>) {
        loop {
            if let Err(err) = run_flush(&self, &batch).await {
                warn!(%err, count = batch.len(), "memory extraction failed; the rows stay pending");
                let mut state = self.state.lock().unwrap();
                // Back to the front, so the retried batch keeps its order
                // around whatever pokes landed meanwhile. Only another full
                // batch with fresh rows continues — a pure retry waits for
                // the next trigger instead of spinning on a broken store.
                let restored = batch.len();
                state.pending.splice(.., batch);
                if state.pending.len() <= restored {
                    return;
                }
            } else if self.state.lock().unwrap().pending.len() < BATCH_ROWS {
                return;
            }
            batch = self.take_pending();
            if batch.is_empty() {
                return;
            }
        }
    }
}

/// One extraction: load the batch's rows, keep the dialogue (user utterances
/// and spoken assistant text), and put one forced decision call through the
/// loop. An error leaves the rows to the next flush — the caller restores
/// them.
async fn run_flush(task: &FlushTask, ids: &[i64]) -> Result<(), String> {
    let rows: Vec<(serde_json::Value, String)> = sqlx::query_as(
        "SELECT message, to_char(created_at, 'YYYY-MM-DD') FROM messages \
         WHERE id = ANY($1) AND user_id = $2 AND agent_id = $3 ORDER BY id",
    )
    .bind(ids.to_vec())
    .bind(task.owner.user_id)
    .bind(task.owner.agent_id)
    .fetch_all(&task.pool)
    .await
    .map_err(|err| err.to_string())?;
    let mut items = Vec::with_capacity(rows.len());
    let mut days: Vec<String> = Vec::new();
    for (value, day) in rows {
        if days.last() != Some(&day) {
            days.push(day);
        }
        match serde_json::from_value(value) {
            Ok(item) => items.push(item),
            Err(err) => warn!(%err, "skipping an unreadable message row"),
        }
    }
    let dialogue = dialogue_lines(&items);
    if dialogue.is_empty() {
        return Ok(());
    }
    let entries = task.entries.list().await;
    let messages = build_messages(&entries, &dialogue, &days);
    let tools = std::slice::from_ref(&task.propose);
    let result = generate_text(
        StreamTextParams::builder(task.llm.as_ref())
            .messages(messages)
            .tools(tools)
            .tool_choice(ToolChoice::Named(MemoryPropose::NAME))
            // A background batch: a little thinking sharpens the distillation
            // and nobody waits on the first token.
            .reasoning(ReasoningEffort::Low)
            .stop_when(vec![stop_after_proposal()])
            .build(),
    )
    .await
    .map_err(|err| err.to_string())?;
    debug!(
        steps = result.steps,
        rows = ids.len(),
        "memory extraction flush finished"
    );
    Ok(())
}

/// The batch's dialogue lines — user utterances and spoken assistant text in
/// order; tool exchange rows are plumbing, not conversation.
#[derive(Debug, PartialEq)]
struct DialogueLine {
    speaker: &'static str,
    text: String,
}

fn dialogue_lines(items: &[ChatItem]) -> Vec<DialogueLine> {
    items
        .iter()
        .filter_map(|item| match item {
            ChatItem::User { content } => Some(DialogueLine {
                speaker: "用户",
                text: content.clone(),
            }),
            ChatItem::Assistant { content, .. } => content
                .as_deref()
                .filter(|text| !text.is_empty())
                .map(|text| DialogueLine {
                    speaker: "助手",
                    text: text.into(),
                }),
            _ => None,
        })
        .collect()
}

/// The flush's one exchange: the job description plus the current entries as
/// the system prompt, the dialogue batch as the user turn — both rendered
/// whole from the module's templates.
fn build_messages(
    entries: &[MemoryEntry],
    dialogue: &[DialogueLine],
    days: &[String],
) -> Vec<ChatItem> {
    let entries = entries
        .iter()
        .map(|entry| {
            context! {
                id => entry_id(entry.mem_no),
                content => &entry.content,
            }
        })
        .collect::<Value>();
    let dialogue = dialogue
        .iter()
        .map(|line| {
            context! {
                speaker => line.speaker,
                text => &line.text,
            }
        })
        .collect::<Value>();
    let system = render_extract(
        "extract_system",
        context! { entries, max_entries => MAX_ENTRIES },
    );
    let user = render_extract(
        "extract_user",
        context! {
            days => days.iter().map(Value::from).collect::<Value>(),
            dialogue,
        },
    );
    vec![ChatItem::system(system), ChatItem::user(user)]
}

fn render_extract(name: &str, ctx: Value) -> String {
    TEMPLATES
        .get_template(name)
        .expect("registered extractor template")
        .render(ctx)
        .expect("valid extractor template")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{InMemEntryMemory, LlmError, LlmEvents, StepResult, ToolCall, ToolSpec};
    use sqlx::postgres::PgPoolOptions;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    #[test]
    fn dialogue_keeps_utterances_and_spoken_replies_only() {
        let items = vec![
            ChatItem::system("prompt"),
            ChatItem::user("你好"),
            ChatItem::assistant_tool_calls(vec![]),
            ChatItem::tool("1", "sunny"),
            ChatItem::assistant("🙂 今天晴"),
        ];
        assert_eq!(
            dialogue_lines(&items),
            vec![
                DialogueLine {
                    speaker: "用户",
                    text: "你好".into(),
                },
                DialogueLine {
                    speaker: "助手",
                    text: "🙂 今天晴".into(),
                },
            ]
        );
    }

    #[test]
    fn messages_carry_the_prompt_entries_and_the_day_span() {
        let entries = vec![MemoryEntry {
            mem_no: 1,
            content: "likes Rust".into(),
        }];
        let dialogue = vec![DialogueLine {
            speaker: "用户",
            text: "你好".into(),
        }];

        let messages = build_messages(&entries, &dialogue, &["2026-10-12".into()]);
        assert_eq!(messages.len(), 2);
        let ChatItem::System { content } = &messages[0] else {
            panic!("expected the extractor prompt")
        };
        assert!(content.starts_with("你是长期记忆提炼器。"));
        assert!(content.contains("memory_propose 工具一次性提交"));
        assert!(content.contains(&format!("条目接近上限（{MAX_ENTRIES}）")));
        assert!(content.ends_with("## 现有记忆条目\n[mem_01] likes Rust\n"));
        let ChatItem::User { content } = &messages[1] else {
            panic!("expected the dialogue batch")
        };
        assert_eq!(content, "对话记录（2026-10-12）：\n\n用户：你好\n");

        let messages = build_messages(&[], &dialogue, &["2026-10-11".into(), "2026-10-12".into()]);
        let ChatItem::System { content } = &messages[0] else {
            panic!("expected the extractor prompt")
        };
        assert!(content.ends_with("## 现有记忆条目\n（暂无）\n"));
        let ChatItem::User { content } = &messages[1] else {
            panic!("expected the dialogue batch")
        };
        assert_eq!(
            content,
            "对话记录（2026-10-11 至 2026-10-12）：\n\n用户：你好\n"
        );
    }

    fn step(tool_calls: Vec<ToolCall>, ok: bool) -> StepResult {
        StepResult {
            tool_calls,
            ok,
            text: String::new(),
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    #[test]
    fn the_stop_condition_needs_a_successful_proposal_round() {
        let condition = stop_after_proposal();
        assert!(!condition(&[]));
        assert!(!condition(&[step(vec![call(MemoryPropose::NAME)], false)]));
        assert!(!condition(&[step(vec![call("memory_add")], true)]));
        assert!(condition(&[step(vec![call(MemoryPropose::NAME)], true)]));
        assert!(condition(&[step(
            vec![call(MemoryPropose::NAME), call(MemoryPropose::NAME)],
            true
        )]));
    }

    struct FailLlm;

    impl Llm for FailLlm {
        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<ToolSpec>,
            _tool_choice: Option<ToolChoice>,
            _reasoning: Option<ReasoningEffort>,
            _cancel: CancellationToken,
        ) -> LlmEvents<'_> {
            Box::pin(futures_util::stream::iter(vec![Err(LlmError::Failed(
                "down".into(),
            ))]))
        }
    }

    fn owner() -> MemoryOwner {
        MemoryOwner {
            user_id: 1,
            agent_id: 2,
            client_id: Uuid::new_v4(),
            device_id: None,
        }
    }

    /// A pool that never connects — the flush's first query fails, which is
    /// the failure the restore path has to survive. The acquire timeout has
    /// to be short: against a machine with nothing listening the pool keeps
    /// retrying the refused connect until the timeout (30s by default),
    /// longer than any deadline below.
    fn lazy_pool() -> PgPool {
        PgPoolOptions::new()
            .acquire_timeout(Duration::from_millis(500))
            .connect_lazy("postgres://localhost:5432/extractor_test_missing")
            .unwrap()
    }

    #[tokio::test]
    async fn pokes_below_the_threshold_only_buffer() {
        let extractor = Extractor::new(
            lazy_pool(),
            owner(),
            Arc::new(FailLlm),
            Arc::new(InMemEntryMemory::default()),
        );
        for i in 0..BATCH_ROWS - 1 {
            extractor.poke(i as i64 + 1);
        }
        assert_eq!(extractor.pending_len(), BATCH_ROWS - 1);
    }

    #[tokio::test]
    async fn a_failed_flush_restores_the_pending_rows() {
        let extractor = Extractor::new(
            lazy_pool(),
            owner(),
            Arc::new(FailLlm),
            Arc::new(InMemEntryMemory::default()),
        );
        for i in 0..BATCH_ROWS {
            extractor.poke(i as i64 + 1);
        }
        // The threshold poke spawns the flush: the buffer drains first, then
        // the failed query restores every row.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while extractor.pending_len() != 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the flush never started"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while extractor.pending_len() != BATCH_ROWS {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the flush never restored the rows"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[cfg(test)]
    impl Extractor {
        pub(crate) fn pending_len(&self) -> usize {
            self.state.lock().unwrap().pending.len()
        }
    }
}
