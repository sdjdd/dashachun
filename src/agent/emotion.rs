use std::fmt;

use unicode_segmentation::UnicodeSegmentation;

macro_rules! emotions {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Emotion {
            $($variant),+
        }

        impl Emotion {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Emotion::$variant => $name),+
                }
            }
        }
    };
}

emotions! {
    Happy => "happy",
    Laughing => "laughing",
    Funny => "funny",
    Loving => "loving",
    Kissy => "kissy",
    Embarrassed => "embarrassed",
    Confident => "confident",
    Cool => "cool",
    Delicious => "delicious",
    Sad => "sad",
    Crying => "crying",
    Sleepy => "sleepy",
    Silly => "silly",
    Angry => "angry",
    Surprised => "surprised",
    Shocked => "shocked",
    Thinking => "thinking",
    Winking => "winking",
    Relaxed => "relaxed",
    Confused => "confused",
    Neutral => "neutral",
}

impl fmt::Display for Emotion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn map(c: char) -> Option<Emotion> {
    use Emotion::*;
    Some(match c {
        '🙂' | '😊' | '☺' => Happy,
        '😀' | '😃' | '😄' | '😁' | '😆' | '🤣' => Laughing,
        '😂' => Funny,
        '😍' | '🥰' | '😻' | '❤' | '💕' => Loving,
        '😘' | '😗' | '😙' | '😚' => Kissy,
        '😳' | '😅' | '😓' => Embarrassed,
        '😎' => Cool,
        '😏' | '😼' => Confident,
        '🤤' | '😋' => Delicious,
        '😔' | '😞' | '😟' | '☹' | '🙁' => Sad,
        '😭' | '😢' | '😥' | '😿' => Crying,
        '😴' | '😪' | '🥱' => Sleepy,
        '😜' | '😝' | '😛' | '🤪' => Silly,
        '😠' | '😡' | '🤬' | '😤' => Angry,
        '😲' | '😮' | '😯' | '😦' | '😧' => Surprised,
        '😱' | '😨' | '😰' | '😵' | '🤯' => Shocked,
        '🤔' | '🧐' | '🤨' => Thinking,
        '😉' => Winking,
        '😌' => Relaxed,
        '🙄' | '😕' | '😖' => Confused,
        '😶' | '😐' | '😑' => Neutral,
        _ => return None,
    })
}

/// The first emoji in `text` that maps to a firmware expression.
pub fn detect(text: &str) -> Option<Emotion> {
    text.chars().find_map(map)
}

/// A grapheme cluster is an emoji when it is a known emoji as a whole. UAX #29
/// groups flag/keycap/ZWJ/skin-tone/tag sequences into one cluster, so a single
/// lookup covers every multi-codepoint sequence. `emojis` stores the emoji-style
/// (VS16) form, so a text-style VS15 sequence is retried without its selector.
fn grapheme_is_emoji(grapheme: &str) -> bool {
    emojis::get(grapheme).is_some()
        || grapheme
            .strip_suffix('\u{FE0E}')
            .is_some_and(|base| emojis::get(base).is_some())
}

/// Removes every emoji from a text stream, tolerating emoji sequences that are
/// split across chunks. Only the final grapheme cluster is held back, since a
/// later chunk may extend it (ZWJ, variation selector, skin tone...); plain text
/// is emitted immediately.
#[derive(Default)]
pub struct Stripper {
    pending: String,
}

impl Stripper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        self.take(false)
    }

    pub fn finish(&mut self) -> String {
        self.take(true)
    }

    fn take(&mut self, eof: bool) -> String {
        let safe = if eof {
            self.pending.len()
        } else {
            self.pending
                .grapheme_indices(true)
                .next_back()
                .map(|(start, _)| start)
                .unwrap_or(0)
        };

        let mut out = String::with_capacity(safe);
        for grapheme in self.pending[..safe].graphemes(true) {
            if !grapheme_is_emoji(grapheme) {
                out.push_str(grapheme);
            }
        }
        self.pending.drain(..safe);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_all(text: &str) -> String {
        let mut stripper = Stripper::new();
        let mut out = stripper.push(text);
        out.push_str(&stripper.finish());
        out
    }

    #[test]
    fn detects_known_emoji() {
        assert_eq!(detect("🙂你好"), Some(Emotion::Happy));
        assert_eq!(detect("😄你好"), Some(Emotion::Laughing));
        assert_eq!(detect("你好😭"), Some(Emotion::Crying));
        assert_eq!(detect("🙄"), Some(Emotion::Confused));
        assert_eq!(detect("😎"), Some(Emotion::Cool));
        assert_eq!(detect("😘"), Some(Emotion::Kissy));
    }

    #[test]
    fn ignores_unknown_and_absent_emoji() {
        assert_eq!(detect("🦄你好"), None);
        assert_eq!(detect("你好"), None);
        assert_eq!(detect(""), None);
    }

    #[test]
    fn strips_emoji_but_keeps_text() {
        assert_eq!(strip_all("🙂你好"), "你好");
        assert_eq!(strip_all("你好😭！"), "你好！");
        assert_eq!(strip_all("no emoji"), "no emoji");
    }

    #[test]
    fn strips_sequences_and_skin_tones() {
        assert_eq!(strip_all("👨‍👩‍👧家"), "家");
        assert_eq!(strip_all("👍🏽好"), "好");
        assert_eq!(strip_all("🇨🇳中国"), "中国");
        assert_eq!(strip_all("中1️⃣文"), "中文");
        assert_eq!(
            strip_all("中🏴󠁧󠁢󠁥󠁮󠁧󠁿文"),
            "中文",
            "tag-sequence flag (England) should be stripped"
        );
    }

    #[test]
    fn strips_text_style_vs15_emoji() {
        assert_eq!(strip_all("中\u{2764}\u{FE0E}文"), "中文");
        assert_eq!(strip_all("中\u{2764}\u{FE0F}文"), "中文");
        assert_eq!(strip_all("中\u{2764}文"), "中文");
    }

    #[test]
    fn keeps_plain_digits_and_symbols() {
        assert_eq!(strip_all("1 # * 42"), "1 # * 42");
        assert_eq!(strip_all("a1b"), "a1b");
    }

    #[test]
    fn holds_trailing_grapheme_until_next_chunk() {
        let mut stripper = Stripper::new();
        assert_eq!(stripper.push("你好🙂"), "你好");
        assert_eq!(stripper.push("再见"), "再");
        assert_eq!(stripper.finish(), "见");
    }

    #[test]
    fn strips_sequence_split_across_chunks() {
        let mut stripper = Stripper::new();
        assert_eq!(stripper.push("👨"), "");
        assert_eq!(stripper.push("\u{200d}"), "");
        assert_eq!(stripper.push("👩\u{200d}👧家"), "");
        assert_eq!(stripper.finish(), "家");
    }

    #[test]
    fn strips_keycap_split_at_ascii_base() {
        let mut stripper = Stripper::new();
        assert_eq!(stripper.push("价1"), "价");
        assert_eq!(stripper.push("\u{FE0F}\u{20E3}中"), "");
        assert_eq!(stripper.finish(), "中");
        assert_eq!(strip_all("价1️⃣中"), "价中");
    }

    #[test]
    fn flushes_plain_text_on_finish() {
        let mut stripper = Stripper::new();
        assert_eq!(stripper.push("你好"), "你");
        assert_eq!(stripper.finish(), "好");
    }

    #[test]
    fn all_names_are_stable() {
        assert_eq!(Emotion::Happy.as_str(), "happy");
        assert_eq!(Emotion::Neutral.as_str(), "neutral");
    }
}
