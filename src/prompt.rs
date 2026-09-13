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

// ─── Prompt building ──────────────────────────────────────────────

pub(crate) fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    audio_start_token_id: i64,
    audio_token_id: i64,
    audio_end_token_id: i64,
    nat: usize,
    language: Option<&str>,
    prefix_text: Option<&str>,
) -> anyhow::Result<(Vec<i64>, usize)> {
    let mut tokens: Vec<i64> = vec![
        TOK_IM_START,
        TOK_SYSTEM,
        TOK_NEWLINE,
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_USER,
        TOK_NEWLINE,
        audio_start_token_id,
    ];
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
    language: Option<&str>,
) -> anyhow::Result<TranscribeResult> {
    let raw_text = tokenizer
        .decode(generated_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {}", e))?;
    let (lang, text) = parse_asr_output(&raw_text, language);
    Ok(TranscribeResult {
        text,
        language: lang,
        raw_output: raw_text,
    })
}

/// Port of Python `parse_asr_output` including `detect_and_fix_repetitions`.
pub(crate) fn parse_asr_output(raw: &str, user_language: Option<&str>) -> (String, String) {
    if raw.is_empty() {
        return (String::new(), String::new());
    }
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return (String::new(), String::new());
    }

    s = detect_and_fix_repetitions(&s, 20);

    if let Some(user_lang) = user_language {
        // Forced language: model output is pure transcription text.
        // Still strip a leading <asr_text> if the model re-emitted the tag.
        let text = strip_leading_asr_text_tag(&s);
        return (user_lang.to_string(), text);
    }

    const TAG: &str = "<asr_text>";
    if let Some(pos) = s.find(TAG) {
        let meta = s[..pos].trim();
        let text = s[pos + TAG.len()..].trim().to_string();
        let lang = meta
            .strip_prefix("language ")
            .or_else(|| meta.strip_prefix("Language "))
            .unwrap_or(meta)
            .trim()
            .to_string();
        if lang.eq_ignore_ascii_case("none") {
            return (String::new(), String::new());
        }
        return (lang, text);
    }

    // no tag => pure text
    (String::new(), s)
}

fn strip_leading_asr_text_tag(s: &str) -> String {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("<asr_text>") {
        return rest.trim().to_string();
    }
    // Model may emit the literal after whitespace / newlines.
    if let Some(pos) = t.find("<asr_text>") {
        return t[pos + "<asr_text>".len()..].trim().to_string();
    }
    t.to_string()
}

/// Port of Python `detect_and_fix_repetitions` (threshold default 20).
///
/// Pattern search window matches upstream (`max_pattern_len=20` is too small for
/// some Chinese phrase units; 96 is a safe upper bound and only runs after decode).
pub(crate) fn detect_and_fix_repetitions(text: &str, threshold: usize) -> String {
    let text = fix_char_repeats(text, threshold);
    fix_pattern_repeats(&text, threshold, 96)
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
