//! Shared pipeline orchestration — Rust port of main.py, used by both the
//! CLI binary and the GUI. Also provides the folder→ZIP batch flow that
//! mirrors gui.py's processing loop.

use std::path::{Path, PathBuf};
use std::time::Instant;

use eat_core::logging;
use eat_core::nlp::{default_language, default_model_name, default_ollama_url, EatError, NlpEngine};
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

pub mod report;

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
    pub api_url: Option<String>,
    /// Bearer token（OpenAI-compatible 端點用；本地 Ollama 免填）
    pub api_key: Option<String>,
    pub ollama_url: Option<String>,
    pub language: Option<String>,
    pub save_intermediate: bool,
    pub no_spell_correct: bool,
    pub no_gpu: bool,
    /// 掃描頁讀取引擎："ppocr"（本機 CRNN，快）或 "vlm"（Ollama 視覺模型
    /// 逐行精讀，手寫準確度大幅提升）。det（行框）一律用本機 PP-OCR。
    pub reader: String,
    pub vlm_model: String,
    /// 評分並產出批改報表 CSV。
    pub score: bool,
    /// 批改標準（文字＋各維度佔比）。預設＝Standard 1/2 的 30 分制。
    pub rubric: report::RubricCfg,
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
                let url = args
                    .api_url
                    .as_deref()
                    .or(args.ollama_url.as_deref())
                    .unwrap_or(&default_ollama_url())
                    .to_string();
                let vlm = eat_ocr::VlmReader::with_key(&vlm_model_name(args), &url, args.api_key.as_deref());
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
    let nlp = NlpEngine::with_key(
        args.model.as_deref().unwrap_or(&default_model_name()),
        args.api_url.as_deref().or(args.ollama_url.as_deref()).unwrap_or(&default_ollama_url()),
        args.api_key.as_deref(),
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
    #[derive(Clone)]
    struct DetailRow {
        essay: String,
        total: f32,
        sentence: String,
        kind: String,
        original: String,
        suggestion: String,
    }
    let mut detail_rows: Vec<DetailRow> = Vec::new();
    #[derive(Clone, Default)]
    struct ScoredEssay {
        scores: Vec<f32>,
        comment: String,
        total: f32,
    }
    let mut score_rows: Vec<(String, ScoredEssay)> = Vec::new();
    let mut pdfs: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x.eq_ignore_ascii_case("pdf")).unwrap_or(false))
                // never re-process our own annotated outputs
                .filter(|p| {
                    !p.file_stem()
                        .map(|s| {
                            let s = s.to_string_lossy();
                            s.ends_with("_annotated") || s.ends_with(".annotated")
                        })
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
                let stem = pdf.file_stem().unwrap().to_string_lossy().to_string();
                let err_list: Vec<EatError> = std::fs::read_to_string(
                    pdf.with_file_name(format!("{}_errors.json", stem)),
                )
                .ok()
                .and_then(|t| serde_json::from_str::<Vec<EatError>>(&t).ok())
                .unwrap_or_default()
                .into_iter()
                .filter(|e| !is_form_noise(&e.original))
                .collect();
                let pages: eat_core::words::Pages = std::fs::read_to_string(
                    pdf.with_file_name(format!("{}_ocr.json", stem)),
                )
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default();
                let language = o.language.clone().unwrap_or_else(|| "en".into());
                let grade = if opts.score && !opts.rubric.dims.is_empty() {
                    let text = pages
                        .iter()
                        .map(|w| eat_core::words::words_to_text(w, &language))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let (scores, comment) = report::grade_with(
                        &opts.rubric,
                        &text,
                        err_list.len(),
                        &o.model.clone().unwrap_or_else(default_model_name),
                        &o.ollama_url.clone().unwrap_or_else(default_ollama_url),
                    );
                    let total: f32 = scores
                        .iter()
                        .zip(opts.rubric.dims.iter())
                        .map(|(s, d)| s.min(d.max))
                        .sum();
                    ScoredEssay { scores, comment, total }
                } else {
                    ScoredEssay::default()
                };
                for (sent, hits) in report::sentence_errors(&pages, &err_list, &language) {
                    for e in hits {
                        detail_rows.push(DetailRow {
                            essay: name.clone(),
                            total: grade.total,
                            sentence: sent.clone(),
                            kind: e.kind.clone(),
                            original: e.original.clone(),
                            suggestion: e.suggestion.clone(),
                        });
                    }
                }
                results.push((name.clone(), err_list.len(), format!("done {:.0}分", grade.total)));
                score_rows.push((name.clone(), grade));
                out_files.push(PathBuf::from(&saved));
                continue;
            }

            Err(e) => {
                logging::error(&format!("{} — {}", name, e));
                results.push((name.clone(), 0, format!("error: {}", e.0)));
            }
        }
    }

    // 批改報表（評分總表＋句級錯誤明細），與標註 PDF 一起進 ZIP
    if opts.score && !score_rows.is_empty() {
        let esc = |t: &str| t.replace('"', "'");
        let dim_header: Vec<String> = opts
            .rubric
            .dims
            .iter()
            .map(|d| format!("{}{}", d.name, d.max))
            .collect();
        let total_max = opts.rubric.total_max();
        let mut csv = format!("essay,{},總分{},錯誤數,短評\n", dim_header.join(","), total_max);
        for (name, g) in &score_rows {
            let errs = detail_rows.iter().filter(|d| d.essay == *name).count();
            let mut row = name.clone();
            for sc in &g.scores {
                row.push_str(&format!(",{}", sc));
            }
            csv.push_str(&format!(
                "{},{},{},\"{}\"\n",
                row,
                g.total,
                errs,
                esc(&g.comment)
            ));
        }
        std::fs::write(std::path::Path::new(input_dir).join(REPORT_SUMMARY), &csv).ok();

        // 每份作文一個明細檔（使用者要求：一個文檔一個錯誤明細）
        for (name, g) in &score_rows {
            let stem = name.trim_end_matches(".pdf");
            let mut dcsv = String::from("總分,問題句子,類型,錯字或錯處,建議\n");
            for d in detail_rows.iter().filter(|d| d.essay == *name) {
                let kind = match d.kind.as_str() {
                    "spelling" => "拼字",
                    "grammar" => "文法",
                    _ => "語意",
                };
                dcsv.push_str(&format!(
                    "{},\"{}\",{},\"{}\",\"{}\"\n",
                    d.total,
                    esc(&d.sentence),
                    kind,
                    esc(&d.original),
                    esc(&d.suggestion)
                ));
            }
            let _ = g; // 分數已顯示於總表；明細行內含總分欄
            std::fs::write(
                std::path::Path::new(input_dir).join(format!("{}_錯誤明細.csv", stem)),
                &dcsv,
            )
            .ok();
        }
        logging::info(&format!(
            "批改報表 → {}（總表＋每份作文一個明細 CSV）",
            std::path::Path::new(input_dir).join(REPORT_SUMMARY).display()
        ));
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
            // 批改報表 CSV 一併打包（總表＋每份作文的明細檔）
            if let Ok(rd) = std::fs::read_dir(input_dir) {
                let mut names: Vec<String> = rd
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n == REPORT_SUMMARY || n.ends_with("_錯誤明細.csv"))
                    .collect();
                names.sort();
                for report_name in names {
                    if let Ok(bytes) = std::fs::read(std::path::Path::new(input_dir).join(&report_name)) {
                        if zip.start_file(report_name.as_str(), zip::write::SimpleFileOptions::default()).is_ok() {
                            let _ = std::io::Write::write_all(&mut zip, &bytes);
                        }
                    }
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

const REPORT_SUMMARY: &str = "批改總表.csv";


/// 試卷表單雜訊判定：頁碼（Page 5 of 7、第3頁、純數字）、點數標示
/// （/20、/10、30 Points）、被數字與標點主導的片段（>60% 字元非字母
/// 且非 CJK）——這些不是學生的寫作錯誤，不該進報表或標註。
pub fn is_form_noise(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    let lower = t.to_lowercase();
    if lower.starts_with("page ") || lower.contains(" page ") || lower.starts_with("第") && lower.ends_with("頁") {
        return true;
    }
    if t.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    let total = t.chars().filter(|c| !c.is_whitespace()).count();
    if total == 0 {
        return true;
    }
    // 數字/標點主導（如 "/20"、"Concluding Sentence /10" 的斜線數字段）
    if (letters as f64 / total as f64) < 0.4 {
        return true;
    }
    // 結尾是 "/數字" 的點數標示
    if let Some(pos) = t.rfind('/') {
        let tail = &t[pos + 1..];
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}
