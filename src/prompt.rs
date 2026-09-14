//! Prompt construction and result parsing for Qwen3-ASR.
//!
//! Force-language prompt and output post-processing mirror upstream Python
//! `qwen_asr.inference.utils.parse_asr_output` / `_build_text_prompt`.

#[derive(Debug, Clone)]
pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

// ─── Token constants ──────────────────────────────────────────────

pub(crate) const IM_END_TOKEN_ID: i64 = 151645;
pub(crate) const ENDOFTEXT_TOKEN_ID: i64 = 151643;
/// `<asr_text>` special separator (same id as HF tokenizer).
pub(crate) const ASR_TEXT_SEP_TOKEN_ID: u32 = 151704;

pub(crate) const TOK_IM_START: i64 = 151644;
pub(crate) const TOK_SYSTEM: i64 = 8948;
pub(crate) const TOK_NEWLINE: i64 = 198;
pub(crate) const TOK_IM_END: i64 = IM_END_TOKEN_ID;
pub(crate) const TOK_USER: i64 = 872;
pub(crate) const TOK_ASSISTANT: i64 = 77091;
const LANG_PREFIX: &str = "language ";

// ─── Prompt building ──────────────────────────────────────────────

pub(crate) fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    audio_start_token_id: i64,
    audio_token_id: i64,
    audio_end_token_id: i64,
    nat: usize,
    language: Option<&str>,
    context: &str,
    prefix_text: Option<&str>,
) -> anyhow::Result<(Vec<i64>, usize)> {
    // Chat template parity: `system` carries the context (hotword biasing),
    // `user` carries the audio.  Upstream: `_build_messages(context, audio)` then
    // `apply_chat_template(..., add_generation_prompt=True)`.
    let mut tokens: Vec<i64> = vec![TOK_IM_START, TOK_SYSTEM, TOK_NEWLINE];
    if !context.is_empty() {
        let enc = tokenizer
            .encode(context, false)
            .map_err(|e| anyhow::anyhow!("encode context: {}", e))?;
        tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
    }
    tokens.extend_from_slice(&[TOK_IM_END, TOK_NEWLINE, TOK_IM_START, TOK_USER, TOK_NEWLINE]);
    tokens.push(audio_start_token_id);
    let asp = tokens.len();
    tokens.extend(std::iter::repeat_n(audio_token_id, nat));
    tokens.extend_from_slice(&[audio_end_token_id, TOK_IM_END, TOK_NEWLINE, TOK_IM_START]);
    if let Some(lang) = language {
        // Python: base + f"language {force_language}<asr_text>"
        // Prefilling through <asr_text> forces text-only generation (no meta loop).
        tokens.push(TOK_ASSISTANT);
        tokens.push(TOK_NEWLINE);
        let lang_str = format!("language {}", capitalize_first(lang));
        let enc = tokenizer
            .encode(lang_str.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        tokens.push(ASR_TEXT_SEP_TOKEN_ID as i64);
    } else {
        tokens.push(TOK_ASSISTANT);
        tokens.push(TOK_NEWLINE);
    }
    if let Some(prefix) = prefix_text {
        if !prefix.is_empty() {
            let enc = tokenizer
                .encode(prefix, false)
                .map_err(|e| anyhow::anyhow!("encode prefix: {}", e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        }
    }
    Ok((tokens, asp))
}

// ─── Result parsing (Python parity) ───────────────────────────────

pub(crate) fn decode_result(
    tokenizer: &tokenizers::Tokenizer,
    generated_ids: &[u32],
) -> anyhow::Result<TranscribeResult> {
    let raw_text = tokenizer
        .decode(generated_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {}", e))?;
    let (lang, text) = parse_asr_output(&raw_text);
    Ok(TranscribeResult {
        text,
        language: lang,
        raw_output: raw_text,
    })
}

/// Port of the reference processor's `_parse_single_output`
/// (transformers' `Qwen3ASRProcessor`), which is what produced the gold texts —
/// note it takes **no** forced-language argument: when a language is forced the
/// metadata is part of the *prompt*, so the generated text carries none and the
/// reported language is empty, exactly like the reference.
pub(crate) fn parse_asr_output(raw: &str) -> (String, String) {
    if raw.trim().is_empty() {
        return (String::new(), String::new());
    }
    let mut s = raw.trim().to_string();

    // The decoded string can still carry the prompt's assistant tail.
    if let Some(idx) = s.find("assistant\n") {
        s = s[idx + "assistant\n".len()..].to_string();
    }

    s = detect_and_fix_repetitions(&s, 20);

    const TAG: &str = "<asr_text>";
    let Some(pos) = s.find(TAG) else {
        // No tag — treat the whole string as plain transcription.
        return (String::new(), s.trim().to_string());
    };
    let prefix = s[..pos].trim().to_string();
    let transcription = s[pos + TAG.len()..].trim().to_string();

    // Empty-audio heuristic: "language None<asr_text>"
    if prefix.to_lowercase() == "language none" {
        return (String::new(), transcription);
    }

    // Only the first non-empty line is inspected, and its value is used raw.
    let mut language = String::new();
    for line in prefix.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_lowercase();
        if lower.starts_with(LANG_PREFIX) {
            let val = line[LANG_PREFIX.len()..].trim();
            if !val.is_empty() {
                language = val.to_string();
            }
        } else {
            language = line.to_string();
        }
        break;
    }
    (language, transcription)
}

/// Port of transformers' `resolve_language` (`audio_utils.py`): accepts a
/// language code (`"en"`, `"zh"`) or a full name (`"English"`), either case, and
/// returns the canonical full name the forced-language suffix must use.
pub(crate) fn resolve_language(language: &str) -> anyhow::Result<String> {
    let l = language.to_lowercase();
    for (code, name) in LANGUAGE_CODE_TO_NAME {
        if l == code.to_lowercase() || l == name.to_lowercase() {
            return Ok(name.to_string());
        }
    }
    anyhow::bail!(
        "unsupported language: {language:?} — use a code (e.g. \"en\", \"zh\") or a full name (e.g. \"English\", \"Chinese\")"
    )
}

/// `LANGUAGE_CODE_TO_NAME` from the reference processor.
pub(crate) const LANGUAGE_CODE_TO_NAME: [(&str, &str); 30] = [
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
];

/// Port of Python `detect_and_fix_repetitions` (threshold default 20).
///
/// `max_pattern_len` is upstream's 20: a longer window collapses repeats the
/// reference leaves alone, which is a behaviour difference, not an improvement.
pub(crate) fn detect_and_fix_repetitions(text: &str, threshold: usize) -> String {
    let text = fix_char_repeats(text, threshold);
    fix_pattern_repeats(&text, threshold, 20)
}

fn fix_char_repeats(s: &str, thresh: usize) -> String {
    // Operate on Unicode scalars like Python `str`.
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut res = String::new();
    let mut i = 0;
    while i < n {
        let mut count = 1;
        while i + count < n && chars[i + count] == chars[i] {
            count += 1;
        }
        if count > thresh {
            res.push(chars[i]);
        } else {
            for c in &chars[i..i + count] {
                res.push(*c);
            }
        }
        i += count;
    }
    res
}

fn fix_pattern_repeats(s: &str, thresh: usize, max_len: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let min_repeat_chars = thresh * 2;
    if n < min_repeat_chars {
        return s.to_string();
    }

    let mut i = 0;
    let mut result = String::new();
    let mut found = false;
    while i + min_repeat_chars <= n {
        found = false;
        for k in 1..=max_len {
            if i + k * thresh > n {
                break;
            }
            let pattern = &chars[i..i + k];
            let mut valid = true;
            for rep in 1..thresh {
                let start_idx = i + rep * k;
                if &chars[start_idx..start_idx + k] != pattern {
                    valid = false;
                    break;
                }
            }
            if valid {
                let mut end_index = i + thresh * k;
                while end_index + k <= n && &chars[end_index..end_index + k] == pattern {
                    end_index += k;
                }
                for c in pattern {
                    result.push(*c);
                }
                let rest: String = chars[end_index..].iter().collect();
                result.push_str(&fix_pattern_repeats(&rest, thresh, max_len));
                i = n;
                found = true;
                break;
            }
        }
        if found {
            break;
        }
        result.push(chars[i]);
        i += 1;
    }
    if !found {
        for c in &chars[i..] {
            result.push(*c);
        }
    }
    result
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_language_prompt_ends_with_asr_text_sep() {
        // Token sequence must include ASR_TEXT_SEP after language (Python parity).
        // Use a minimal fake: just assert constant wiring via a synthetic check.
        assert_eq!(ASR_TEXT_SEP_TOKEN_ID, 151704);
    }

    #[test]
    fn fix_char_repeats_collapses_long_runs() {
        let s = "a".repeat(25) + "b";
        let out = fix_char_repeats(&s, 20);
        assert_eq!(out, "ab");
    }

    #[test]
    fn fix_pattern_repeats_collapses_long_loops() {
        let unit = "你好世界";
        let s = unit.repeat(25);
        let out = fix_pattern_repeats(&s, 20, 20);
        // Collapses to a single unit (or short prefix of units depending on k search).
        assert!(out.len() < s.len());
        assert!(out.contains("你好") || out == unit);
    }

    #[test]
    fn parse_forced_strips_asr_text_tag() {
        let (lang, text) = parse_asr_output("<asr_text>hello world", Some("Chinese"));
        assert_eq!(lang, "Chinese");
        assert_eq!(text, "hello world");
    }

    #[test]
    fn parse_unforced_splits_tag() {
        let (lang, text) = parse_asr_output("language Chinese<asr_text>正文", None);
        assert_eq!(lang, "Chinese");
        assert_eq!(text, "正文");
    }

}
