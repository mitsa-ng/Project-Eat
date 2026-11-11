fn main() {
    let dir = "/tmp/rubric-test";
    let zip = "/tmp/rubric-test/out.zip";
    let rubric: eat_app::report::RubricCfg = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("HOME").unwrap() + "/eat-rubric.json").unwrap(),
    ).unwrap();
    println!("滿分 = {}", rubric.total_max());
    let opts = eat_app::PipelineOpts {
        save_intermediate: true,
        no_spell_correct: true,
        reader: "vlm".into(),
        vlm_model: "qwen2.5vl:7b".into(),
        score: true,
        rubric,
        ..Default::default()
    };
    let (_z, r) = eat_app::run_batch(dir, zip, &opts);
    for (n, e, s) in r { println!("{} — {} errors — {}", n, e, s); }
}
