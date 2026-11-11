//! 依 Standard 1/2 評分規準為作文打分（重新審閱）。
//! 用法：score <ocr-json-dir>... <output.csv>
//! 讀取每份 *_ocr.json（前次批改保留的 OCR 文字）＋對應 *_errors.json，
//! 以規準 prompt 評分，輸出 CSV 總表。

use std::time::Instant;

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

#[derive(serde::Deserialize, Debug, Clone)]
struct Word {
    text: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct ErrItem {
    original: String,
    suggestion: String,
}

#[derive(serde::Deserialize, Debug, Clone, Default)]
struct Score {
    #[serde(default)]
    topic_sentence: f32,
    #[serde(default)]
    supporting_unity: f32,
    #[serde(default)]
    grammar_diction: f32,
    #[serde(default)]
    concluding: f32,
    #[serde(default)]
    total: f32,
    #[serde(default)]
    comment: String,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_csv = args.last().expect("usage: score <dir>... <out.csv>").clone();
    let dirs = &args[..args.len() - 1];
    assert!(!dirs.is_empty(), "at least one dir required");

    let mut rows: Vec<(String, Score, usize)> = Vec::new();
    for dir in dirs {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().ends_with("_ocr.json"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        for f in files {
            let stem = f.file_stem().unwrap().to_string_lossy();
            let stem = stem.strip_suffix("_ocr").unwrap_or(&stem).to_string();
            let ocr: Vec<Vec<Word>> = serde_json::from_str(&std::fs::read_to_string(&f).unwrap())
                .unwrap_or_default();
            let text: String = ocr
                .iter()
                .flat_map(|p| p.iter())
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let err_path = f.with_file_name(format!("{}_errors.json", stem));
            let err_count = std::fs::read_to_string(&err_path)
                .ok()
                .and_then(|s| serde_json::from_str::<Vec<ErrItem>>(&s).ok())
                .map(|v| v.len())
                .unwrap_or(0);
            if text.trim().is_empty() {
                continue;
            }
            let t0 = Instant::now();
            let score = grade(&text, err_count);
            eprintln!(
                "{}  {}  total={:.0}  [{}]  {:.0}s",
                stem,
                if score.total > 0.0 { "OK " } else { "???" },
                score.total,
                score.comment,
                t0.elapsed().as_secs()
            );
            rows.push((stem, score, err_count));
        }
    }

    let mut csv = String::from("essay,topic_6,support_8,grammar_10,concl_6,total_30,errors,comment\n");
    for (stem, s, e) in &rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},\"{}\"\n",
            stem, s.topic_sentence, s.supporting_unity, s.grammar_diction, s.concluding, s.total, e,
            s.comment.replace('"', "'")
        ));
    }
    std::fs::write(&out_csv, &csv).unwrap();
    println!("\n{}", csv);
    println!("報表 → {}", out_csv);
}

fn grade(text: &str, err_count: usize) -> Score {
    let model = eat_core::nlp::default_model_name();
    let url = eat_core::nlp::default_ollama_url();
    let agent = ureq::AgentBuilder::new()
        .timeout_read(std::time::Duration::from_secs(300))
        .build();
    let prompt = format!(
        "Detected language errors in this essay ({} items): see below the essay.\n\nESSAY OCR TEXT:\n{}\n\n(An automatic checker additionally found {} spelling/grammar issues in this essay.)",
        0, text, err_count
    );
    let payload = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "system": RUBRIC,
        "stream": false,
        "options": {"temperature": 0, "num_predict": 512}
    });
    for _ in 0..2 {
        if let Ok(resp) = agent.post(&url).timeout(std::time::Duration::from_secs(300)).send_json(payload.clone()) {
            if let Ok(v) = resp.into_json::<serde_json::Value>() {
                let raw = v.get("response").and_then(|r| r.as_str()).unwrap_or("");
                if let Some(s) = raw.find('{') {
                    if let Some(e) = raw.rfind('}') {
                        if let Ok(sc) = serde_json::from_str::<Score>(&raw[s..=e]) {
                            return sc;
                        }
                    }
                }
            }
        }
    }
    Score::default()
}
