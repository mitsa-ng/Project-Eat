//! Shared pipeline orchestration — Rust port of main.py, used by both the
//! CLI binary and the GUI. Also provides the folder→ZIP batch flow that
//! mirrors gui.py's processing loop.

use std::path::{Path, PathBuf};
use std::time::Instant;

use eat_core::logging;
use eat_core::nlp::{default_language, default_model_name, default_ollama_url, NlpEngine};
use eat_core::words::Word;
use eat_ocr::imgops::Img;
use eat_ocr::RapidOcr;
use eat_pdf::{Annotator, EatPdf};
use std::sync::{Mutex, OnceLock};

/// pdfium's dynamic binding initializes exactly once per process; every
/// pipeline call shares this instance (serialized behind a Mutex).
static PDF_ENGINE: OnceLock<Option<EatPdf>> = OnceLock::new();
/// PP-OCR sessions are expensive to load (~30 MB of weights); share them too.
static OCR_ENGINE: OnceLock<Option<Mutex<RapidOcr>>> = OnceLock::new();

pub fn shared_pdf() -> Result<&'static EatPdf, PipelineError> {
    PDF_ENGINE
        .get_or_init(|| EatPdf::bind().ok())
        .as_ref()
        .ok_or_else(|| PipelineError("pdfium bind failed on first init".into()))
}

pub fn shared_ocr() -> Result<&'static Mutex<RapidOcr>, PipelineError> {
    if let Some(o) = OCR_ENGINE.get() {
        return o.as_ref().ok_or_else(|| PipelineError("PP-OCR models failed to load on first init".into()));
    }
    let loaded = RapidOcr::load().ok().map(|m| Mutex::new(m));
    let stored = OCR_ENGINE.get_or_init(|| loaded);
    stored.as_ref().ok_or_else(|| PipelineError("PP-OCR models failed to load (EAT_PPROCR_DIR / models/ beside executable / vendor/ppocr)".into()))
}

#[derive(Debug, Clone, Default)]
pub struct PipelineOpts {
    pub input_pdf: Option<String>,
    pub output: Option<String>,
    pub model: Option<String>,
    pub ollama_url: Option<String>,
    pub language: Option<String>,
    pub save_intermediate: bool,
    pub no_spell_correct: bool,
    pub no_gpu: bool,
    /// 掃描頁讀取引擎："ppocr"（本機 CRNN，快）或 "vlm"（Ollama 視覺模型
    /// 逐行精讀，手寫準確度大幅提升）。det（行框）一律用本機 PP-OCR。
    pub reader: String,
    pub vlm_model: String,
}

fn vlm_model_name(opts: &PipelineOpts) -> String {
    if opts.vlm_model.is_empty() { "qwen2.5vl:7b".into() } else { opts.vlm_model.clone() }
}

#[derive(Debug)]
pub struct PipelineError(pub String);

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
    // resolve symlinks via the (existing) parent, like Python's Path.resolve()
    let resolved = if let (Some(parent), Some(name)) = (resolved.parent(), resolved.file_name()) {
        let cp = parent.canonicalize().unwrap_or_else(|_| parent.to_path_buf());
        cp.join(name)
    } else {
        resolved
    };
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

pub fn run_pipeline(args: &PipelineOpts) -> Result<String, PipelineError> {
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
    logging::info("  Digital PDF → pdfium  |  Scanned/Handwritten → PP-OCR (Rust)");
    let pdf = shared_pdf()?;
    let t1 = Instant::now();
    let extracts = pdf
        .extract_pdf(&inp.to_string_lossy())
        .map_err(|e| PipelineError(e.to_string()))?;
    let mut pages: Vec<Vec<Word>> = Vec::new();
    const SCAN_DPI: f32 = 300.0;
    const CONFIDENCE_THRESHOLD: f32 = 0.4;
    for (i, ex) in extracts.iter().enumerate() {
        if ex.scanned {
            let t = Instant::now();
            let ocr_lock = shared_ocr()?;
            let mut engine = ocr_lock.lock().unwrap();
            let (w, h, rgb) = pdf
                .render_page_rgb(&inp.to_string_lossy(), i as i32, SCAN_DPI)
                .map_err(|e| PipelineError(e.to_string()))?;
            let img = Img::from_rgb(w as usize, h as usize, &rgb);
            let lines = if args.reader == "vlm" {
                let url = args.ollama_url.clone().unwrap_or_else(default_ollama_url);
                let vlm = eat_ocr::VlmReader::new(&vlm_model_name(args), &url);
                logging::info(&format!("  Page {}: VLM 行讀取（{}）…", i + 1, vlm_model_name(args)));
                engine.run_with_reader(&img, &vlm).map_err(|e| PipelineError(e.to_string()))?
            } else {
                engine.run(&img).map_err(|e| PipelineError(e.to_string()))?
            };
            // px → PDF points (300 dpi render): pt = px * 72 / dpi
            let scale = 72.0 / SCAN_DPI as f64;
            let mut words = Vec::new();
            for l in lines {
                if l.text.trim().is_empty() || l.score < CONFIDENCE_THRESHOLD {
                    continue;
                }
                words.push(Word {
                    text: l.text.clone(),
                    box_: [
                        [l.box_[0].0 * scale, l.box_[0].1 * scale],
                        [l.box_[1].0 * scale, l.box_[1].1 * scale],
                        [l.box_[2].0 * scale, l.box_[2].1 * scale],
                        [l.box_[3].0 * scale, l.box_[3].1 * scale],
                    ],
                });
            }
            logging::info(&format!(
                "  Page {}: SCAN/HANDWRITTEN  ({} words via PP-OCR @ {} DPI, {:.1}s)",
                i + 1,
                words.len(),
                SCAN_DPI,
                t.elapsed().as_secs_f32()
            ));
            pages.push(words);
        } else {
            logging::info(&format!(
                "  Page {}: DIGITAL  ({} words via pdfium)",
                i + 1,
                ex.words.len()
            ));
            pages.push(ex.words.clone());
        }
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
            logging::info(&format!(
                "  {:3}. [{:8}] {:?} → {:?}",
                i + 1,
                e.kind,
                e.original,
                e.suggestion
            ));
        }
    }

    // OCR lines carry multi-word text; split into proportional word boxes so
    // the annotator's span matcher can place tags on scanned essays.
    let pages = eat_pdf::split_line_words(&pages);

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


/// Process every PDF in `input_dir` and pack the annotated outputs into a
/// single ZIP (port of gui.py's batch loop). Returns (zip_path, per-file
/// (name, error_count, status_string)) pairs for the results table.
pub fn run_batch(
    input_dir: &str,
    zip_path: &str,
    opts: &PipelineOpts,
) -> (String, Vec<(String, usize, String)>) {
    let mut results: Vec<(String, usize, String)> = Vec::new();
    let mut pdfs: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x.eq_ignore_ascii_case("pdf")).unwrap_or(false))
                // never re-process our own annotated outputs
                .filter(|p| {
                    !p.file_stem()
                        .map(|s| s.to_string_lossy().ends_with("_annotated"))
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    pdfs.sort();
    if pdfs.is_empty() {
        logging::error(&format!("No PDFs found in {}", input_dir));
        return (zip_path.to_string(), results);
    }
    logging::info(&format!("Processing {} PDF(s) from {}", pdfs.len(), input_dir));

    let mut out_files: Vec<PathBuf> = Vec::new();
    for (i, pdf) in pdfs.iter().enumerate() {
        let name = pdf.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        logging::info(&format!("File {}/{} — {}", i + 1, pdfs.len(), name));
        let out = pdf.with_extension("annotated.pdf");
        let mut o = opts.clone();
        o.input_pdf = Some(pdf.to_string_lossy().into_owned());
        o.output = Some(out.to_string_lossy().into_owned());
        match run_pipeline(&o) {
            Ok(saved) => {
                // count errors from the run log? simpler: ask pipeline —
                // run_pipeline returns the path; error count comes from the
                // intermediate json when requested. For the results table we
                // re-read the saved errors json when save_intermediate.
                let errs = o
                    .save_intermediate
                    .then(|| {
                        pdf.with_file_name(pdf.file_stem().unwrap().to_string_lossy().to_string() + "_errors.json")
                    })
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.as_array().map(|a| a.len()))
                    .unwrap_or(0);
                results.push((name.clone(), errs, "done".into()));
                out_files.push(PathBuf::from(&saved));
            }
            Err(e) => {
                logging::error(&format!("{} — {}", name, e));
                results.push((name.clone(), 0, format!("error: {}", e.0)));
            }
        }
    }

    // pack outputs into the ZIP
    let zf = std::fs::File::create(zip_path);
    match zf {
        Ok(f) => {
            let mut zip = zip::ZipWriter::new(f);
            let mut ok = true;
            for out in &out_files {
                let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                match std::fs::read(out) {
                    Ok(bytes) => {
                        if zip.start_file(name.as_str(), zip::write::SimpleFileOptions::default()).is_err()
                            || std::io::Write::write_all(&mut zip, &bytes).is_err()
                        {
                            ok = false;
                        }
                    }
                    Err(_) => ok = false,
                }
            }
            if zip.finish().is_err() {
                ok = false;
            }
            if ok {
                logging::info(&format!("Output ZIP → {} ({} file(s))", zip_path, out_files.len()));
            } else {
                logging::error("Failed writing output ZIP");
            }
        }
        Err(e) => logging::error(&format!("Cannot create {}: {}", zip_path, e)),
    }
    (zip_path.to_string(), results)
}
