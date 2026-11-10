//! eat — pipeline orchestrator (Rust port of main.py)
//! Runs: extract (pdfium digital path) → NLPEngine (Ollama) → Annotator
//!
//! Usage:
//!     eat essay.pdf
//!     eat essay.pdf -o corrected.pdf --save-intermediate
//!     eat essay.pdf --model phi3 --language zh-Hant

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use eat_core::languages::LANGUAGES;
use eat_core::logging;
use eat_core::nlp::{default_language, default_model_name, default_ollama_url, NlpEngine};
use eat_core::words::Word;
use eat_pdf::{Annotator, EatPdf};

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

    /// Keep raw OCR words (no auto spell-correction). Digital pages are
    /// unaffected; the scanned-page OCR path arrives in phase 2.
    #[arg(long)]
    no_spell_correct: bool,

    /// Accepted for Python-CLI compatibility; pdfium needs no GPU flag.
    #[arg(long)]
    no_gpu: bool,
}

#[derive(Debug)]
struct PipelineError(String);

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Port of main.py _sanitize_output_path: keep the output inside the input
/// file's directory.
fn sanitize_output_path(output: &str, input_dir: &Path) -> Result<PathBuf, PipelineError> {
    let out = Path::new(output);
    // join, then make absolute against the CWD like Python's Path.resolve()
    let joined = if out.is_absolute() {
        out.to_path_buf()
    } else {
        let j = input_dir.join(out);
        if j.is_absolute() {
            j
        } else {
            std::env::current_dir()
                .unwrap_or_default()
                .join(j)
        }
    };
    // lexical normalize (no fs access; output may not exist yet)
    let resolved = joined
        .components()
        .fold(PathBuf::new(), |mut acc, c| {
            match c {
                std::path::Component::ParentDir => {
                    acc.pop();
                }
                std::path::Component::CurDir => {}
                other => acc.push(other.as_os_str()),
            }
            acc
        });
    let input_resolved = input_dir.canonicalize().unwrap_or_else(|_| {
        std::env::current_dir()
            .unwrap_or_default()
            .join(input_dir)
    });
    if resolved != input_resolved && !resolved.starts_with(&input_resolved) {
        return Err(PipelineError(format!(
            "Output path escapes input directory: {:?} not under {:?}",
            resolved, input_resolved
        )));
    }
    Ok(resolved)
}

fn run_pipeline(args: &Args) -> Result<String, PipelineError> {
    let t0 = Instant::now();
    let input = args
        .input_pdf
        .as_deref()
        .ok_or_else(|| PipelineError("No input PDF given.  Usage: eat essay.pdf".into()))?;
    let inp = PathBuf::from(input);
    if !inp.exists() {
        return Err(PipelineError(format!("File not found: {}", inp.display())));
    }

    let stem = inp.file_stem().and_then(|s| s.to_str()).unwrap_or("essay");
    let default_out = format!("{}_annotated.pdf", stem);
    let out_pdf = sanitize_output_path(
        args.output.as_deref().unwrap_or(&default_out),
        inp.parent().unwrap_or(Path::new(".")),
    )?;

    // ── Step 1/3 — extraction ──────────────────────────────────────────────
    logging::info("━━━ Step 1/3 — OCR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    logging::info("  Digital PDF → pdfium  |  Scanned/Handwritten → phase 2");
    let pdf = EatPdf::bind().map_err(|e| PipelineError(e.to_string()))?;
    let t1 = Instant::now();
    let extracts = pdf
        .extract_pdf(&inp.to_string_lossy())
        .map_err(|e| PipelineError(e.to_string()))?;
    let mut pages: Vec<Vec<Word>> = Vec::new();
    for (i, ex) in extracts.iter().enumerate() {
        if ex.scanned {
            logging::warn(&format!(
                "  Page {}: SCAN/HANDWRITTEN — scanned-page OCR is phase 2 of the Rust port; \
                 this page contributes 0 words",
                i + 1
            ));
        } else {
            logging::info(&format!(
                "  Page {}: DIGITAL  ({} words via pdfium)",
                i + 1,
                ex.words.len()
            ));
        }
        pages.push(ex.words.clone());
    }
    let total_words: usize = pages.iter().map(|p| p.len()).sum();
    logging::info(&format!(
        "OCR done: {} words  |  {} page(s)  [{:.1}s]",
        total_words,
        pages.len(),
        t1.elapsed().as_secs_f32()
    ));
    if total_words == 0 {
        return Err(PipelineError(
            "OCR found 0 words. Check that the PDF is readable and not corrupted.".into(),
        ));
    }

    // ── Step 2/3 — NLP ─────────────────────────────────────────────────────
    logging::info("━━━ Step 2/3 — Grammar Analysis (LLM) ━━━━━━━━━━━━━━━━━━━━━━");
    let nlp = NlpEngine::new(
        args.model.as_deref().unwrap_or(&default_model_name()),
        args.ollama_url.as_deref().unwrap_or(&default_ollama_url()),
        args.language.as_deref().unwrap_or(&default_language()),
    );
    let t2 = Instant::now();
    let errors = nlp
        .analyse_pages(&pages)
        .map_err(|e| PipelineError(e.to_string()))?;
    logging::info(&format!(
        "Analysis done: {} error(s)  [{:.1}s]",
        errors.len(),
        t2.elapsed().as_secs_f32()
    ));

    if errors.is_empty() {
        logging::warn(
            "No errors returned by the model.  Possible causes:\n  \
             • Ollama not running  →  run:  ollama serve\n  \
             • Model not pulled   →  run:  ollama pull qwen2.5:7b\n  \
             • OCR text garbled   →  check _ocr.json with --save-intermediate\n  \
             Annotated PDF will still be saved with legend only.",
        );
    } else {
        for (i, e) in errors.iter().enumerate() {
            let pg = match e.page {
                Some(p) => format!("p{} ", p + 1),
                None => String::new(),
            };
            logging::info(&format!(
                "  {:3}. {}[{:8}] {:?} → {:?}",
                i + 1,
                pg,
                e.kind,
                e.original,
                e.suggestion
            ));
        }
    }

    // ── Step 3/3 — annotate ────────────────────────────────────────────────
    logging::info("━━━ Step 3/3 — Annotating PDF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let ann = Annotator::new(args.language.as_deref().unwrap_or(&default_language()));
    let result = ann
        .annotate_pdf(
            pdf.pdfium(),
            &inp.to_string_lossy(),
            &pages,
            &errors,
            &out_pdf.to_string_lossy(),
        )
        .map_err(|e| PipelineError(e.to_string()))?;
    let _ = result;
    logging::info(&format!(
        "Annotated PDF saved → {}  [{:.1}s]",
        out_pdf.display(),
        t0.elapsed().as_secs_f32()
    ));

    // ── Optional: intermediate files ───────────────────────────────────────
    if args.save_intermediate {
        let ocr_path = inp.with_file_name(format!("{}_ocr.json", stem));
        let err_path = inp.with_file_name(format!("{}_errors.json", stem));
        let ocr_json = serde_json::to_string_pretty(&pages)
            .map_err(|e| PipelineError(e.to_string()))?;
        let err_json = serde_json::to_string_pretty(&errors)
            .map_err(|e| PipelineError(e.to_string()))?;
        std::fs::write(&ocr_path, ocr_json)
            .map_err(|e| PipelineError(format!("write {}: {}", ocr_path.display(), e)))?;
        std::fs::write(&err_path, err_json)
            .map_err(|e| PipelineError(format!("write {}: {}", err_path.display(), e)))?;
        logging::info(&format!("OCR data   → {}", ocr_path.display()));
        logging::info(&format!("Error list → {}", err_path.display()));
    }

    logging::info(&format!(
        "━━━ Pipeline complete in {:.1}s ━━━━━━━━━━━━━━━━━━━",
        t0.elapsed().as_secs_f32()
    ));
    Ok(out_pdf.to_string_lossy().into_owned())
}

fn main() {
    let args = Args::parse();
    if args.no_gpu || args.no_spell_correct {
        logging::debug("flag accepted for Python-CLI compatibility (no-op in phase 1)");
    }
    match run_pipeline(&args) {
        Ok(path) => {
            let _ = path;
        }
        Err(e) => {
            logging::error(&e.to_string());
            std::process::exit(1);
        }
    }
}
