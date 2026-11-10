//! Verify the GUI's batch core: folder of PDFs → annotated outputs → ZIP.
fn main() {
    let dir = std::env::args().nth(1).expect("usage: batch <folder-with-pdfs> <out.zip>");
    let zip = std::env::args().nth(2).expect("usage: batch <folder> <out.zip>");
    let opts = eat_app::PipelineOpts {
        save_intermediate: true,
        no_spell_correct: true,
        ..Default::default()
    };
    let (zip, results) = eat_app::run_batch(&dir, &zip, &opts);
    println!("ZIP: {}", zip);
    for (name, n, status) in results {
        println!("  {} — {} errors — {}", name, n, status);
    }
}
