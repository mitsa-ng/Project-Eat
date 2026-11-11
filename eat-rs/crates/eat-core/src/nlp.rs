//! NLP analysis module — port of nlp_engine.py.
//!
//! Sends essay text to a local LLM via Ollama for proofreading, with:
//!   • chunked processing for long essays (avoids context overflow)
//!   • 4-tier fault-tolerant JSON parser (wrapped / embedded / line-by-line)
//!   • validation + deduplication + hallucination filtering

use std::collections::HashSet;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::languages::{self, get_system_prompt, is_no_space, normalize};
use crate::logging;
use crate::words::{words_to_text, Pages};

// ── Ollama config (env-overridable, mirroring nlp_engine.py) ───────────────
pub const TEMPERATURE: f64 = 0.0;
pub const MAX_TOKENS: u64 = 4096;
pub const CHUNK_WORDS: usize = 200; // space-delimited essays
pub const CHUNK_CHARS: usize = 350; // non-space scripts (zh/ja)

pub fn default_ollama_url() -> String {
    std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/generate".to_string())
}
pub fn default_model_name() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string())
}
pub fn default_language() -> String {
    std::env::var("EAT_LANGUAGE")
        .unwrap_or_else(|_| languages::DEFAULT_LANGUAGE.to_string())
}

/// One detected error. JSON shape is identical to the Python pipeline's
/// `_errors.json` (field "type" holds spelling|grammar|semantic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EatError {
    pub original: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub suggestion: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
}

#[derive(Debug)]
pub enum NlpError {
    OllamaUnreachable(String),
    ModelNotFound(String),
    Http(String),
}

impl std::fmt::Display for NlpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NlpError::OllamaUnreachable(m) => write!(f, "{}", m),
            NlpError::ModelNotFound(m) => write!(f, "{}", m),
            NlpError::Http(m) => write!(f, "{}", m),
        }
    }
}

pub struct NlpEngine {
    pub model: String,
    pub ollama_url: String,
    pub api_key: Option<String>,
    pub language: String,
    system_prompt: &'static str,
    agent: ureq::Agent,
}

impl NlpEngine {
    pub fn new(model: &str, ollama_url: &str, language: &str) -> Self {
        Self::with_key(model, ollama_url, None, language)
    }

    pub fn with_key(model: &str, ollama_url: &str, api_key: Option<&str>, language: &str) -> Self {
        let language = normalize(language).to_string();
        NlpEngine {
            model: model.to_string(),
            ollama_url: ollama_url.to_string(),
            api_key: api_key.map(|s| s.to_string()),
            system_prompt: get_system_prompt(&language),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(30))
                .timeout_read(Duration::from_secs(600))
                .build(),
            language,
        }
    }

    /// 偵測 API 類型：URL 含 /api/generate → Ollama；否則視為
    /// OpenAI-compatible（/v1/chat/completions）。
    fn is_ollama(&self) -> bool {
        self.ollama_url.contains("/api/generate")
    }

    /// 組請求（Ollama 或 OpenAI 格式）
    fn build_request(&self, text: &str) -> serde_json::Value {
        if self.is_ollama() {
            serde_json::json!({
                "model": self.model,
                "prompt": text,
                "system": self.system_prompt,
                "stream": true,
                "options": { "temperature": TEMPERATURE, "num_predict": MAX_TOKENS }
            })
        } else {
            // OpenAI-compatible chat format
            let url = self.openai_chat_url();
            serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS as u32,
                "stream": true
            })
        }
    }

    fn openai_chat_url(&self) -> String {
        let base = self.ollama_url
            .trim_end_matches('/')
            .trim_end_matches("/chat/completions")
            .trim_end_matches("/v1");
        format!("{}/v1/chat/completions", base)
    }

    /// 通用端點健康檢查：嘗試列出模型。回傳 (可達, 模型列表, 錯誤訊息)。
    /// Ollama → /api/tags；OpenAI-compatible → /v1/models（帶 Bearer）。
    pub fn check_endpoint(url: &str, api_key: Option<&str>) -> (bool, Vec<String>, String) {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout_read(Duration::from_secs(10))
            .build();

        // 嘗試 Ollama
        if url.contains("/api/generate") || url.contains("11434") {
            let tags_url = url.replace("/api/generate", "/api/tags");
            if let Ok(resp) = agent.get(&tags_url).call() {
                if let Ok(v) = resp.into_json::<serde_json::Value>() {
                    let models: Vec<String> = v
                        .get("models")
                        .and_then(|m| m.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    return (true, models, "Ollama".into());
                }
            }
            return (false, vec![], format!("Ollama 端點無回應：{}", tags_url));
        }

        // 嘗試 OpenAI-compatible
        let base = url
            .trim_end_matches('/')
            .trim_end_matches("/chat/completions")
            .trim_end_matches("/generate")
            .trim_end_matches("/v1");
        let models_url = format!("{}/v1/models", base);
        let mut req = agent.get(&models_url);
        if let Some(key) = api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }
        match req.call() {
            Ok(resp) => {
                if resp.status() == 401 {
                    return (false, vec![], "API key 無效或過期（401 Unauthorized）".into());
                }
                if let Ok(v) = resp.into_json::<serde_json::Value>() {
                    let models: Vec<String> = v
                        .get("data")
                        .and_then(|d| d.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m.get("id").and_then(|n| n.as_str()).map(|s| s.to_string()))
                                .take(50)
                                .collect()
                        })
                        .unwrap_or_default();
                    return (true, models, "OpenAI-compatible".into());
                }
                (true, vec![], "已連線（無法解析模型列表）".into())
            }
            Err(ureq::Error::Status(code, _)) => {
                let msg = match code {
                    401 => "API key 無效或過期",
                    403 => "API key 權限不足",
                    404 => "端點不存在（404）",
                    _ => &format!("HTTP {}", code),
                };
                (false, vec![], msg.to_string())
            }
            Err(_) => (false, vec![], format!("無法連線：{}", models_url)),
        }
    }

    // ── chunking ──────────────────────────────────────────────────────────────

    /// Split text into overlapping sentence-boundary chunks.
    /// Port of `NLPEngine._chunk_text` (Python regex lookbehind replaced by a
    /// manual scanner: split after `.!?` + whitespace, or after `。！？`).
    pub fn chunk_text(text: &str, language: &str) -> Vec<String> {
        let no_space = is_no_space(language);
        let max_units = if no_space { CHUNK_CHARS } else { CHUNK_WORDS };
        let sep = if no_space { "" } else { " " };
        let size = |s: &str| {
            if no_space {
                s.chars().count()
            } else {
                s.split_whitespace().count()
            }
        };

        let sentences = split_sentences(text);
        let mut chunks: Vec<String> = Vec::new();
        let mut current: Vec<&str> = Vec::new();
        let mut current_units = 0usize;

        for sent in &sentences {
            let u = size(sent);
            if current_units + u > max_units && !current.is_empty() {
                chunks.push(current.join(sep));
                // keep last sentence as overlap context
                let last = *current.last().unwrap();
                current = vec![last, sent.as_str()];
                current_units = current.iter().map(|s| size(s)).sum();
            } else {
                current.push(sent.as_str());
                current_units += u;
            }
        }
        if !current.is_empty() {
            chunks.push(current.join(sep));
        }
        if chunks.is_empty() {
            vec![text.to_string()]
        } else {
            chunks
        }
    }

    // ── Ollama call ───────────────────────────────────────────────────────────

    /// POST to Ollama with streaming + loop detection, returning the full
    /// accumulated response text. Retries once on transient failures.
    fn call_ollama(&self, text: &str, chunk_index: usize, total_chunks: usize) -> Result<String, NlpError> {
        let word_count = text.split_whitespace().count();
        logging::info(&format!(
            "  Calling Ollama chunk {}/{} ({} words, model={}) …",
            chunk_index, total_chunks, word_count, self.model
        ));
        let payload = self.build_request(text);
        let request_url = if self.is_ollama() {
            self.ollama_url.clone()
        } else {
            self.openai_chat_url()
        };

        let mut last_err: Option<String> = None;
        for attempt in 1..=2 {
            let mut req = self
                .agent
                .post(&request_url)
                .timeout(Duration::from_secs(600));
            if let Some(key) = &self.api_key {
                req = req.set("Authorization", &format!("Bearer {}", key));
            }
            match req
                .send_json(serde_json::to_value(&payload).unwrap())
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status >= 400 {
                        if status == 404 {
                            return Err(NlpError::ModelNotFound(format!(
                                "Ollama model '{}' not found.  Pull it first:  ollama pull {}",
                                self.model, self.model
                            )));
                        }
                        if status >= 500 && attempt == 1 {
                            logging::warn(&format!("  Ollama server error {} — retrying in 2s …", status));
                            std::thread::sleep(Duration::from_secs(2));
                            continue;
                        }
                        return Err(NlpError::Http(format!("Ollama HTTP {}", status)));
                    }
                    return Ok(self.read_stream(resp, word_count));
                }
                Err(ureq::Error::Status(code, _)) => {
                    if code == 404 {
                        return Err(NlpError::ModelNotFound(format!(
                            "Ollama model '{}' not found.  Pull it first:  ollama pull {}",
                            self.model, self.model
                        )));
                    }
                    if code >= 500 && attempt == 1 {
                        logging::warn(&format!("  Ollama server error {} — retrying in 2s …", code));
                        std::thread::sleep(Duration::from_secs(2));
                        continue;
                    }
                    return Err(NlpError::Http(format!("Ollama HTTP {}", code)));
                }
                Err(ureq::Error::Transport(t)) => {
                    let msg = format!("Cannot reach Ollama.  Start it first:  ollama serve\nThen pull the model:  ollama pull {}", self.model);
                    let is_connection = matches!(
                        t.kind(),
                        ureq::ErrorKind::ConnectionFailed | ureq::ErrorKind::Dns
                    );
                    if attempt == 1 && is_connection {
                        logging::warn("  Ollama connection dropped — retrying in 2s …");
                        last_err = Some(msg);
                        std::thread::sleep(Duration::from_secs(2));
                        continue;
                    }
                    return Err(NlpError::OllamaUnreachable(msg));
                }
            }
        }
        Err(NlpError::OllamaUnreachable(
            last_err.unwrap_or_else(|| "Ollama request failed after retry".to_string()),
        ))
    }

    /// Read a streaming Ollama response, accumulating `response` chunks.
    /// Small local models occasionally loop on noisy OCR text, repeating the
    /// same error objects indefinitely (observed: >10 min per chunk). We cut
    /// the stream as soon as an `original` value appears a 3rd time or the
    /// object count becomes implausible; the tier-3 line parser then
    /// recovers every complete object from the accumulated prefix.
    fn read_stream(&self, resp: ureq::Response, word_count: usize) -> String {
        use std::collections::HashMap;
        use std::io::BufRead;

        let max_objects = (word_count / 2).max(40) as u32;
        let mut reader = std::io::BufReader::new(resp.into_reader());
        let mut raw = String::new();
        let mut seen: HashMap<String, u32> = HashMap::new();
        let mut line = String::new();
        let mut lines_since_check = 0u32;

        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF — model finished
                Ok(_) => {}
                Err(_) => break,
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let chunk: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(t) = chunk.get("response").and_then(|v| v.as_str()) {
                raw.push_str(t);
            }
            if chunk.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                break;
            }

            // loop detection over complete "original":"…" spans. Counts are
            // rebuilt from scratch each check so every occurrence is counted
            // exactly once (an incremental map double-counts as raw grows).
            lines_since_check += 1;
            if lines_since_check >= 8 {
                lines_since_check = 0;
                seen.clear();
                let mut count = 0u32;
                let mut looping = false;
                let mut rest = raw.as_str();
                while let Some(pos) = rest.find("\"original\"") {
                    let after = &rest[pos + 10..];
                    let val_start = after.find('"').map(|i| i + 1);
                    if let Some(vs) = val_start {
                        if let Some(vend) = after[vs..].find('"') {
                            count += 1;
                            let val = &after[vs..vs + vend];
                            let e = seen.entry(val.to_string()).or_insert(0);
                            *e += 1;
                            if *e >= 3 {
                                looping = true;
                            }
                            rest = &after[vs + vend + 1..];
                            continue;
                        }
                    }
                    break; // incomplete trailing span — wait for more tokens
                }
                if looping || count > max_objects {
                    logging::warn(&format!(
                        "  Model response is looping — cutting stream, parsing prefix (count={} max={})",
                        count, max_objects
                    ));
                    break;
                }
            }
        }
        logging::info(&format!("  Ollama responded ({} chars)", raw.chars().count()));
        raw
    }

    // ── fault-tolerant JSON parser (4 tiers) ─────────────────────────────────

    /// Extract a JSON array from the model response. Port of `NLPEngine._parse`.
    pub fn parse(raw: &str) -> Vec<Value> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Vec::new();
        }

        // Tier 1: direct parse
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            match v {
                Value::Array(a) => return a,
                Value::Object(map) => {
                    for key in ["errors", "results", "corrections", "issues", "data"] {
                        if let Some(Value::Array(a)) = map.get(key) {
                            return a.clone();
                        }
                    }
                    if map.contains_key("original") {
                        return vec![Value::Object(map)];
                    }
                    // flatten all list values
                    return map
                        .into_values()
                        .filter_map(|v| match v {
                            Value::Array(a) => Some(a),
                            _ => None,
                        })
                        .flatten()
                        .collect();
                }
                _ => {} // scalar: fall through to Tier 2 (same as Python)
            }
        }

        // Tier 2: extract outermost [...] block (greedy: first '[' to last ']')
        if let (Some(s), Some(e)) = (raw.find('['), raw.rfind(']')) {
            if e > s {
                if let Ok(Value::Array(a)) = serde_json::from_str(&raw[s..=e]) {
                    return a;
                }
            }
        }

        // Tier 3: line-by-line objects
        let mut items = Vec::new();
        for line in raw.lines() {
            let line = line.trim().trim_end_matches(',');
            if line.starts_with('{') && line.ends_with('}') {
                if let Ok(Value::Object(map)) = serde_json::from_str(line) {
                    if map.contains_key("original") {
                        items.push(Value::Object(map));
                    }
                }
            }
        }
        if !items.is_empty() {
            return items;
        }

        // Tier 4: collapsed object with list-as-string values.
        // (Unreachable for valid-JSON objects since Tier 1 returns first;
        // ported verbatim for parity with nlp_engine.py.)
        if let Ok(Value::Object(collapsed)) = serde_json::from_str::<Value>(raw) {
            let to_list = |val: &Value| -> Vec<String> {
                match val {
                    Value::Array(a) => a
                        .iter()
                        .map(|v| v.to_string().trim().to_string())
                        .collect(),
                    other => extract_quoted_spans(&other.to_string()),
                }
            };
            let originals = to_list(collapsed.get("original").unwrap_or(&Value::Null));
            let types = to_list(collapsed.get("type").unwrap_or(&Value::Null));
            let suggestions = to_list(collapsed.get("suggestion").unwrap_or(&Value::Null));
            if !originals.is_empty() {
                return originals
                    .into_iter()
                    .enumerate()
                    .map(|(idx, orig)| {
                        let kind = types.get(idx).cloned().unwrap_or_else(|| "grammar".to_string());
                        let sugg = suggestions.get(idx).cloned().unwrap_or_default();
                        json!({
                            "original":   orig,
                            "type":       kind,
                            "suggestion": sugg,
                        })
                    })
                    .collect();
            }
        }

        logging::warn("Could not parse model response — skipping chunk.");
        logging::debug(&format!("  Raw: {}", &raw[..raw.len().min(400)]));
        Vec::new()
    }

    // ── validation / filtering ────────────────────────────────────────────────

    /// Normalise and deduplicate error entries. Port of `NLPEngine._validate`.
    pub fn validate(errors: Vec<Value>, language: &str) -> Vec<EatError> {
        let no_space = is_no_space(language);
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();

        // Expand top-level nested lists into a flat sequence
        let mut flat: Vec<Value> = Vec::new();
        for item in errors {
            match &item {
                Value::Array(inner) => {
                    if matches!(inner.first(), Some(Value::Object(_))) {
                        flat.extend(inner.iter().cloned());
                    } else {
                        flat.push(item);
                    }
                }
                _ => flat.push(item),
            }
        }

        for item in flat {
            // coerce list -> dict
            let item = match item {
                Value::Array(elements) => {
                    let strings: Vec<String> = elements
                        .into_iter()
                        .filter_map(|el| match el {
                            Value::String(s) => Some(s.trim().to_string()),
                            Value::Number(n) => Some(n.to_string()),
                            Value::Bool(b) => Some(b.to_string()),
                            _ => None, // Python keeps str(None-ish); scalars cover real cases
                        })
                        .collect();
                    match strings.len() {
                        n if n >= 3 => Value::Object(
                            [
                                ("original".to_string(), Value::String(strings[0].clone())),
                                ("type".to_string(), Value::String(strings[1].clone())),
                                ("suggestion".to_string(), Value::String(strings[2].clone())),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                        2 => Value::Object(
                            [
                                ("original".to_string(), Value::String(strings[0].clone())),
                                ("type".to_string(), Value::String("grammar".to_string())),
                                ("suggestion".to_string(), Value::String(strings[1].clone())),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                        _ => continue,
                    }
                }
                other => other,
            };

            let map = match item {
                Value::Object(m) => m,
                _ => continue,
            };

            let original = string_field(&map, "original");
            let mut kind = string_field(&map, "type").to_lowercase();
            let suggestion = string_field(&map, "suggestion");

            if original.is_empty() {
                continue;
            }
            // Reject sentences masquerading as errors (model hallucination).
            let too_long = if no_space {
                original.chars().count() > 12
            } else {
                original.split_whitespace().count() > 6
            };
            if too_long {
                logging::debug(&format!("  Skipping over-long error: {:?}", original));
                continue;
            }
            if kind != "spelling" && kind != "grammar" && kind != "semantic" {
                kind = "grammar".to_string();
            }
            if !seen.insert(original.clone()) {
                continue;
            }

            out.push(EatError {
                original,
                kind,
                suggestion,
                page: None,
            });
        }
        out
    }

    /// Remove errors where suggestion is identical to original (no-op).
    /// Port of `NLPEngine._filter_useless`.
    pub fn filter_useless(errors: Vec<EatError>) -> Vec<EatError> {
        let mut kept = Vec::new();
        for e in errors {
            if e.original.to_lowercase() == e.suggestion.to_lowercase() {
                logging::debug(&format!("  Filtered no-op: {:?} -> {:?}", e.original, e.suggestion));
                continue;
            }
            kept.push(e);
        }
        kept
    }

    /// Remove errors whose 'original' is not verbatim in the essay text.
    /// Port of `NLPEngine._filter_hallucinations`.
    pub fn filter_hallucinations(errors: Vec<EatError>, text: &str, language: &str) -> Vec<EatError> {
        if errors.is_empty() {
            return Vec::new();
        }
        let no_space = is_no_space(language);
        let norm = |s: &str| -> String {
            let lower = s.to_lowercase();
            if no_space {
                lower.chars().filter(|c| !c.is_whitespace()).collect()
            } else {
                lower
            }
        };
        let text_cmp = norm(text);
        errors
            .into_iter()
            .filter(|e| {
                let ok = !e.original.is_empty() && text_cmp.contains(&norm(&e.original));
                if !ok {
                    logging::debug(&format!(
                        "  Filtered hallucination: {:?} not in essay text",
                        e.original
                    ));
                }
                ok
            })
            .collect()
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Proofread a plain text string (chunked, merged, deduplicated).
    pub fn analyse_text(&self, text: &str) -> Result<Vec<EatError>, NlpError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let chunks = Self::chunk_text(text, &self.language);
        let mut all_err: Vec<Value> = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            logging::info(&format!(
                "  Analysing chunk {}/{} ({} words) …",
                i + 1,
                chunks.len(),
                chunk.split_whitespace().count()
            ));
            let raw = self.call_ollama(chunk, i + 1, chunks.len())?;
            let errors = Self::parse(&raw);
            logging::info(&format!("  Chunk {}/{}: {} error(s)", i + 1, chunks.len(), errors.len()));
            all_err.extend(errors);
        }

        let final_ = Self::validate(all_err, &self.language);
        let final_ = Self::filter_hallucinations(final_, text, &self.language);
        let final_ = Self::filter_useless(final_);
        logging::info(&format!("Analysis complete: {} error(s)", final_.len()));
        Ok(final_)
    }

    /// Proofread from OCR output. Each page is analysed separately and every
    /// error is tagged with its page number.
    pub fn analyse_pages(&self, pages: &Pages) -> Result<Vec<EatError>, NlpError> {
        let page_texts: Vec<String> = pages
            .iter()
            .map(|words| words_to_text(words, &self.language).trim().to_string())
            .collect();
        if page_texts.iter().all(|t| t.is_empty()) {
            logging::warn("Empty OCR text — nothing to analyse.");
            return Ok(Vec::new());
        }

        let mut all_errors = Vec::new();
        for (pg, page_text) in page_texts.iter().enumerate() {
            if page_text.is_empty() {
                continue;
            }
            logging::info(&format!(
                "Analysing page {} ({} words) …",
                pg + 1,
                page_text.split_whitespace().count()
            ));
            let errors = self.analyse_text(page_text)?;
            for mut e in errors {
                e.page = Some(pg as u32);
                all_errors.push(e);
            }
        }
        Ok(all_errors)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Python `re.split(r'(?<=[.!?])\s+|(?<=[。！？])\s*', text)` — split after
/// `.!?` when followed by whitespace (whitespace consumed), or after `。！？`
/// with any following whitespace consumed.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut sentences = Vec::new();
    let mut start = 0usize;

    let mut i = 0usize;
    while i < chars.len() {
        let (byte_pos, c) = chars[i];
        match c {
            '.' | '!' | '?' => {
                // split only if followed by whitespace (which is consumed)
                if let Some(&(_, next)) = chars.get(i + 1) {
                    if next.is_whitespace() {
                        let end = byte_pos + c.len_utf8();
                        sentences.push(&text[start..end]);
                        // consume whitespace run
                        let mut j = i + 1;
                        while let Some(&(_, nc)) = chars.get(j) {
                            if nc.is_whitespace() {
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        if j < chars.len() {
                            start = chars[j].0;
                        } else {
                            start = text.len();
                        }
                        i = j;
                        continue;
                    }
                }
            }
            '。' | '！' | '？' => {
                let end = byte_pos + c.len_utf8();
                sentences.push(&text[start..end]);
                let mut j = i + 1;
                while let Some(&(_, nc)) = chars.get(j) {
                    if nc.is_whitespace() {
                        j += 1;
                    } else {
                        break;
                    }
                }
                if j < chars.len() {
                    start = chars[j].0;
                } else {
                    start = text.len();
                }
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        sentences.push(&text[start..]);
    }
    sentences
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// str() of a JSON object field: String as-is, numbers/bools stringified,
/// missing/null → "". (Python `str(item.get(k, "") or "")`.)
fn string_field(map: &serde_json::Map<String, Value>, key: &str) -> String {
    match map.get(key) {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// Extract all '...' and "..." spans from a string (Python `re.findall(r"[\'\"](.*?)[\'\"]", s)`).
fn extract_quoted_spans(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' {
            let mut span = String::new();
            let mut closed = false;
            for nc in chars.by_ref() {
                if nc == c {
                    closed = true;
                    break;
                }
                span.push(nc);
            }
            if closed {
                out.push(span);
            }
        }
    }
    out
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_english() {
        let s = split_sentences("She go to school. He likes cats! Do you? Yes.");
        assert_eq!(s, vec!["She go to school.", "He likes cats!", "Do you?", "Yes."]);
    }

    #[test]
    fn split_sentences_keeps_decimals() {
        let s = split_sentences("Pi is 3.14 ok. Next sentence.");
        assert_eq!(s, vec!["Pi is 3.14 ok.", "Next sentence."]);
    }

    #[test]
    fn split_sentences_cjk() {
        let s = split_sentences("她去學校。他很喜歡貓！真的嗎？");
        assert_eq!(s, vec!["她去學校。", "他很喜歡貓！", "真的嗎？"]);
    }

    #[test]
    fn chunk_text_respects_word_budget() {
        let text = "One two three four. ".repeat(60); // 240 words > CHUNK_WORDS
        let chunks = NlpEngine::chunk_text(&text, "en");
        assert!(chunks.len() > 1, "expected multiple chunks, got {}", chunks.len());
        for c in &chunks {
            assert!(c.split_whitespace().count() <= CHUNK_WORDS + 20);
        }
    }

    #[test]
    fn chunk_text_cjk_joins_without_space() {
        let text = "這是一個句子。".repeat(100);
        let chunks = NlpEngine::chunk_text(&text, "zh-Hant");
        assert!(!chunks.is_empty());
        // no-space scripts join without separators
        assert!(!chunks[0].contains(' '));
        // sentences keep their trailing 。 punctuation
        assert!(chunks[0].contains('。'));
    }

    #[test]
    fn parse_tier1_array() {
        let raw = r#"[{"original":"intresting","type":"spelling","suggestion":"interesting"}]"#;
        let v = NlpEngine::parse(raw);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn parse_tier1_dict_wrapper() {
        let raw = r#"{"errors": [{"original":"a","type":"spelling","suggestion":"b"}]}"#;
        assert_eq!(NlpEngine::parse(raw).len(), 1);
    }

    #[test]
    fn parse_tier2_embedded() {
        let raw = "Sure! Here you go:\n[{\"original\":\"a\",\"type\":\"grammar\",\"suggestion\":\"b\"}]\nDone.";
        assert_eq!(NlpEngine::parse(raw).len(), 1);
    }

    #[test]
    fn parse_tier3_line_by_line() {
        let raw = "{\"original\":\"a\",\"type\":\"spelling\",\"suggestion\":\"b\"},\n{\"original\":\"c\",\"type\":\"grammar\",\"suggestion\":\"d\"},";
        let v = NlpEngine::parse(raw);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn validate_coerces_list_shapes() {
        let raw = r#"[
            ["intresting", "spelling", "interesting"],
            ["she go", "she goes"],
            {"original":"x y","type":"semantic","suggestion":"y x"}
        ]"#;
        let out = NlpEngine::validate(NlpEngine::parse(raw), "en");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].kind, "spelling");
        assert_eq!(out[1].kind, "grammar"); // inferred for 2-element shape
    }

    #[test]
    fn validate_dedupes_and_caps_length() {
        let raw = r#"[
            {"original":"recieved","type":"spelling","suggestion":"received"},
            {"original":"recieved","type":"spelling","suggestion":"received"},
            {"original":"this sentence is way way too long here","type":"grammar","suggestion":"x"}
        ]"#;
        let out = NlpEngine::validate(NlpEngine::parse(raw), "en");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn validate_caps_cjk_by_chars() {
        let long = "錯".repeat(13);
        let raw = format!(r#"[{{"original":"{}","type":"spelling","suggestion":"x"}}]"#, long);
        assert!(NlpEngine::validate(NlpEngine::parse(&raw), "zh-Hant").is_empty());
    }

    #[test]
    fn filter_hallucinations_no_space_matching() {
        let text = "我 去 學 校 上 課 。"; // OCR token-joining introduces gaps
        let err = EatError {
            original: "去學校".to_string(),
            kind: "grammar".to_string(),
            suggestion: "去上学".to_string(),
            page: None,
        };
        let kept = NlpEngine::filter_hallucinations(vec![err], text, "zh-Hant");
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn filter_useless_removes_noops() {
        let err = EatError {
            original: "Word".to_string(),
            kind: "grammar".to_string(),
            suggestion: "word".to_string(),
            page: None,
        };
        assert!(NlpEngine::filter_useless(vec![err]).is_empty());
    }

    #[test]
    fn words_to_text_joins_by_language() {
        use crate::words::Word;
        let mk = |t: &str| Word {
            text: t.to_string(),
            box_: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let ws = vec![mk("她去"), mk("學校")];
        assert_eq!(words_to_text(&ws, "zh-Hant"), "她去學校");
        let ws = vec![mk("she"), mk("goes")];
        assert_eq!(words_to_text(&ws, "en"), "she goes");
    }
}
