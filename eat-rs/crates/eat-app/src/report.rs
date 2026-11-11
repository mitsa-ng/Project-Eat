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

/// 規準評分（Standard 1/2 重建，滿分 30）。只評學生手寫段落，忽略
/// 試卷表單文字（說明、題目、點數、頁碼）。
pub fn grade(text: &str, err_count: usize, model: &str, url: &str) -> Grade {
    const RUBRIC: &str = r#"You are grading an in-class English paragraph writing exam (English Composition III Midterm, PART 4) using the official grading rubric below. Grade ONLY the student's handwritten paragraph; ignore exam-form furniture in the OCR text (instructions, prompts, rubric wording, point values, page numbers).

GRADING RUBRIC (30 points total):
- Topic Sentence (6 pts): clear topic + controlling idea.
- Supporting Sentences & Unity (8 pts): 5-6+ detailed sentences with specific details/facts/examples; strictly on topic; no fragments; correct count/non-count nouns; clear pronouns; formal tone.
- Grammar & Academic Diction (10 pts): grammar correctness, spelling, academic word choice.
- Concluding Sentence (6 pts): logical conclusion/restatement using an appropriate transition signal (e.g., Overall, In conclusion, Therefore).

WRITING REQUIREMENTS: paragraph of at least 8 sentences (approx. 100-150 words).

OUTPUT: ONLY a JSON object, no other text:
{"topic_sentence": <0-6>, "supporting_unity": <0-8>, "grammar_diction": <0-10>, "concluding": <0-6>, "total": <0-30>, "comment": "<繁體中文 40-60 字短評：優點與最大改進方向>"}
"#;
    let payload = serde_json::json!({
        "model": model,
        "prompt": format!("ESSAY OCR TEXT:\n{}\n\n(An automatic checker found {} spelling/grammar issues in this essay.)", text, err_count),
        "system": RUBRIC,
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
                    if let Ok(g) = serde_json::from_str::<Grade>(&raw[a..=b]) {
                        return g;
                    }
                }
            }
        }
    }
    Grade::default()
}

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
