//! 批改報表：句級錯誤對應（哪一句有問題、拼錯字在哪）＋規準評分整合。

use eat_core::nlp::EatError;
use eat_core::words::{words_to_text, Pages};

/// 把錯誤對應到包含它的句子。回傳 (句子, 屬於該句的錯誤)。
/// 對不上的錯誤（OCR 差異）附在最後一條「未定位」。
pub fn sentence_errors<'a>(pages: &Pages, errors: &'a [EatError], language: &str) -> Vec<(String, Vec<&'a EatError>)> {
    let text = pages
        .iter()
        .map(|w| words_to_text(w, language))
        .collect::<Vec<_>>()
        .join(" ");
    let sentences = eat_core::nlp::split_sentences(&text);
    let mut out: Vec<(String, Vec<&EatError>)> = Vec::new();
    let mut placed = vec![false; errors.len()];
    for sent in &sentences {
        let lower = sent.to_lowercase();
        // 句子層表單過濾：頁碼條、規準文字、點數標示不是學生寫作
        if is_form_sentence(&lower) {
            continue;
        }
        let mut hits: Vec<&EatError> = Vec::new();
        for (i, e) in errors.iter().enumerate() {
            if lower.contains(&e.original.to_lowercase()) {
                hits.push(e);
                placed[i] = true;
            }
        }
        if !hits.is_empty() {
            out.push((sent.clone(), hits));
        }
    }
    let unplaced: Vec<&EatError> = errors
        .iter()
        .enumerate()
        .filter(|(i, _)| !placed[*i])
        .map(|(_, e)| e)
        .collect();
    if !unplaced.is_empty() {
        out.push(("（未定位）".to_string(), unplaced));
    }
    out
}

/// 可設定批改規準：評分說明文字＋各維度佔比（名稱與滿分）。
/// 預設＝Standard 1/2 重建的 30 分制（主題句6／支持句8／文法10／結論6）。
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct RubricCfg {
    /// 給模型的評分說明（批改標準本文）
    pub criteria: String,
    /// 評分維度（名稱, 滿分）。滿分可自由調整，總分＝各維度之和。
    pub dims: Vec<(String, f32)>,
}

impl Default for RubricCfg {
    fn default() -> Self {
        RubricCfg {
            criteria: DEFAULT_CRITERIA.to_string(),
            dims: vec![
                ("主題句".into(), 6.0),
                ("支持句".into(), 8.0),
                ("文法".into(), 10.0),
                ("結論句".into(), 6.0),
            ],
        }
    }
}

pub const DEFAULT_CRITERIA: &str = r#"Grade the student's handwritten paragraph (ignore exam-form furniture: instructions, prompts, point values, page numbers).
- 主題句 Topic Sentence: clear topic + controlling idea.
- 支持句 Supporting Sentences & Unity: 5-6+ detailed sentences with specific details/facts/examples; strictly on topic; no fragments; correct count/non-count nouns; clear pronouns; formal tone.
- 文法 Grammar & Academic Diction: grammar correctness, spelling, academic word choice.
- 結論句 Concluding Sentence: logical conclusion/restatement using an appropriate transition signal (e.g., Overall, In conclusion, Therefore).
Writing requirement: at least 8 sentences (approx. 100-150 words)."#;

impl RubricCfg {
    pub fn total_max(&self) -> f32 {
        self.dims.iter().map(|(_, m)| *m).sum()
    }
}

#[derive(serde::Deserialize, Debug, Clone, Default)]
struct RawGrade {
    #[serde(default)]
    d: Vec<f32>,
    #[serde(default)]
    comment: String,
}

/// 依規準評分。回傳各維度得分（順序同 dims）＋短評。
pub fn grade_with(rubric: &RubricCfg, text: &str, err_count: usize, model: &str, url: &str) -> (Vec<f32>, String) {
    let n = rubric.dims.len();
    let dim_lines: Vec<String> = rubric
        .dims
        .iter()
        .enumerate()
        .map(|(i, (name, max))| format!("- d{} \"{}\": 0-{} points", i, name, max))
        .collect();
    let keys: Vec<String> = (0..n).map(|i| format!("\"d{}\": <0-max>", i)).collect();
    let system = format!(
        "You are grading a student essay paragraph using the teacher's rubric below.\n\nGRADING CRITERIA:\n{}\n\nDIMENSIONS:\n{}\n\nOUTPUT: ONLY a JSON object, no other text:\n{{{}, \"comment\": \"<繁體中文 40-60 字短評：優點與最大改進方向>\"}}\nScore each dimension honestly; do not inflate.",
        rubric.criteria,
        dim_lines.join("\n"),
        keys.join(", ")
    );
    let payload = serde_json::json!({
        "model": model,
        "prompt": format!("ESSAY OCR TEXT:\n{}\n\n(An automatic checker found {} spelling/grammar issues.)", text, err_count),
        "system": system,
        "stream": false,
        "options": {"temperature": 0, "num_predict": 512}
    });
    let agent = ureq::AgentBuilder::new()
        .timeout_read(std::time::Duration::from_secs(300))
        .build();
    for _ in 0..2 {
        if let Ok(resp) = agent
            .post(url)
            .timeout(std::time::Duration::from_secs(300))
            .send_json(payload.clone())
        {
            if let Ok(v) = resp.into_json::<serde_json::Value>() {
                let raw = v.get("response").and_then(|r| r.as_str()).unwrap_or("");
                if let (Some(a), Some(b)) = (raw.find('{'), raw.rfind('}')) {
                    if let Ok(g) = serde_json::from_str::<serde_json::Value>(&raw[a..=b]) {
                        // 抓 d0..dN（也可能放在巢狀）
                        let mut scores = Vec::with_capacity(n);
                        for i in 0..n {
                            let key = format!("d{}", i);
                            let val = g.get(&key).and_then(|x| x.as_f64()).or_else(|| {
                                g.get("scores").and_then(|s| s.get(&key)).and_then(|x| x.as_f64())
                            });
                            scores.push(val.unwrap_or(0.0) as f32);
                        }
                        let comment = g
                            .get("comment")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string();
                        return (scores, comment);
                    }
                }
            }
        }
    }
    (vec![0.0; n], String::new())
}

// ── 舊介面相容（沿用 default 規準）────────────────────────────────────────

#[derive(serde::Deserialize, Debug, Clone, Default)]
pub struct Grade {
    #[serde(default)]
    pub topic_sentence: f32,
    #[serde(default)]
    pub supporting_unity: f32,
    #[serde(default)]
    pub grammar_diction: f32,
    #[serde(default)]
    pub concluding: f32,
    #[serde(default)]
    pub total: f32,
    #[serde(default)]
    pub comment: String,
}

/// 句子是否為試卷表單文字（頁碼條 / 規準描述 / 點數）。
fn is_form_sentence(lower: &str) -> bool {
    // "page 5 of 7" 頁碼條
    if lower.contains("page ") && lower.contains(" of ") {
        let words: Vec<&str> = lower.split_whitespace().collect();
        for w in words {
            if w == "page" {
                continue;
            }
        }
        return lower.matches("page").count() >= 1
            && lower.split_whitespace().any(|w| w.parse::<u32>().is_ok());
    }
    // 規準文字特徵
    for marker in [
        "logical conclusion/restatement",
        "transition signal",
        "detailed sentences; specific examples",
        "grading rubric",
        "controlling idea",
    ] {
        if lower.contains(marker) {
            return true;
        }
    }
    false
}
