//! eat — CLI wrapper; the pipeline itself lives in eat-app (shared with GUI).
//!
//! Usage:
//!     eat essay.pdf
//!     eat essay.pdf -o corrected.pdf --save-intermediate
//!     eat essay.pdf --model qwen2.5:7b --language zh-Hant

use clap::Parser;

use eat_app::PipelineOpts;
#[derive(Parser, Debug)]
#[command(
    name = "eat",
    about = "AI Essay Corrector — hybrid OCR + local-LLM grammar analysis with per-language grading rubrics",
    version
)]
struct Args {
    /// Path to the essay PDF (digital or scanned/handwritten)
    input_pdf: Option<String>,

    /// Output annotated PDF (default: <input>_annotated.pdf)
    #[arg(short, long)]
    output: Option<String>,

    /// Ollama model name (default: qwen2.5:7b, override with env OLLAMA_MODEL)
    #[arg(long)]
    model: Option<String>,

    /// Ollama endpoint URL (override with env OLLAMA_URL)
    #[arg(long)]
    ollama_url: Option<String>,

    /// Essay language / grading rubric (override with env EAT_LANGUAGE)
    #[arg(short, long)]
    language: Option<String>,

    /// Also save _ocr.json and _errors.json alongside output
    #[arg(long)]
    save_intermediate: bool,

    /// Keep raw OCR words (no auto spell-correction; the Rust port never
    /// spell-corrects — the LLM sees genuine misspellings by design).
    #[arg(long)]
    no_spell_correct: bool,

    /// Accepted for Python-CLI compatibility; pdfium needs no GPU flag.
    #[arg(long)]
    no_gpu: bool,

    /// Scanned-page reading engine: "ppocr" (local CRNN, fast, default) or
    /// "vlm" (Ollama vision model reads each detected line — far better on
    /// handwriting).
    #[arg(long, default_value = "ppocr")]
    reader: String,

    /// Vision model for --reader vlm (default: qwen2.5vl:7b, needs
    /// `ollama pull qwen2.5vl:7b`).
    #[arg(long, default_value = "qwen2.5vl:7b")]
    vlm_model: String,

    /// 依 Standard 1/2 規則評分（滿分 30）。批次模式另產出批改總表＋
    /// 錯誤明細 CSV。
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    score: bool,
}

fn main() {
    let args = Args::parse();
    let opts = PipelineOpts {
        input_pdf: args.input_pdf,
        output: args.output,
        model: args.model,
        ollama_url: args.ollama_url,
        language: args.language,
        save_intermediate: args.save_intermediate,
        no_spell_correct: args.no_spell_correct,
        no_gpu: args.no_gpu,
        reader: args.reader,
        vlm_model: args.vlm_model,
        score: args.score,
        rubric: eat_app::report::RubricCfg::default(),
    };
    if let Err(e) = eat_app::run_pipeline(&opts) {
        eat_core::logging::error(&e.0);
        std::process::exit(1);
    }
}
