//! Verify the GUI's batch core: folder of PDFs → annotated outputs → ZIP.
fn main() {
    let dir = std::env::args().nth(1).expect("usage: batch <folder-with-pdfs> <out.zip>");
    let zip = std::env::args().nth(2).expect("usage: batch <folder> <out.zip>");
    let reader = std::env::args().nth(3).unwrap_or_else(|| "ppocr".into());
    let score = std::env::args().nth(4).map(|v| v != "no").unwrap_or(true);
    let rubric = std::fs::read_to_string(
        std::env::var("HOME").unwrap_or_default() + "/eat-rubric.json",
    )
    .ok()
    .and_then(|t| serde_json::from_str(&t).ok())
    .unwrap_or_default();
    let opts = eat_app::PipelineOpts {
        save_intermediate: true,
        no_spell_correct: true,
        reader,
        vlm_model: "qwen2.5vl:7b".into(),
        score,
        rubric,
        ..Default::default()
    };
    let (zip, results) = eat_app::run_batch(&dir, &zip, &opts);
    println!("ZIP: {}", zip);
    for (name, n, status) in results {
        println!("  {} — {} errors — {}", name, n, status);
    }
}
