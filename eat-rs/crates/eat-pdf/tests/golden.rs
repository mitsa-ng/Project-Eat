//! Integration tests against the golden fixtures captured from the Python
//! pipeline (pdfs/standard-test*.pdf/json).

use eat_core::nlp::EatError;
use eat_core::words::Pages;
use eat_pdf::{Annotator, EatPdf};
use std::collections::BTreeMap;

/// Python pipeline's _ocr.json shape: {"0": [ {text, box}, … ], "1": […]}
fn load_golden_ocr() -> Pages {
    let raw =
        std::fs::read_to_string(fixtures_dir().join("standard-test_ocr.json")).unwrap();
    let map: BTreeMap<u32, Vec<eat_core::words::Word>> =
        serde_json::from_str(&raw).unwrap();
    let max_page = map.keys().copied().max().unwrap_or(0);
    let mut pages: Pages = Vec::new();
    for i in 0..=max_page {
        pages.push(map.get(&i).cloned().unwrap_or_default());
    }
    pages
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .expect("fixtures dir exists")
}

fn bind() -> EatPdf {
    let lib = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/pdfium-macos-arm64/lib/libpdfium.dylib");
    std::env::set_var("EAT_PDFIUM_LIB", lib.canonicalize().unwrap());
    // bind_to_library can only run once per process; a second test reuses the
    // already-initialized binding through the same candidate path.
    EatPdf::bind().expect("pdfium binds")
}

#[test]
fn extracts_digital_words_from_standard_test() {
    let pdf = bind();
    let extracts = pdf
        .extract_pdf(&fixtures_dir().join("standard-test.pdf").to_string_lossy())
        .expect("extract");
    assert!(!extracts.is_empty());
    assert!(
        !extracts[0].scanned,
        "standard-test.pdf is a digital PDF and must not be classified as scanned"
    );
    let n = extracts[0].words.len();
    assert!(n >= 15, "expected a digital page to yield ≥15 words, got {}", n);

    // sanity: boxes must be sane top-left coordinates inside a US-Letter page
    for w in &extracts[0].words {
        let (x0, y0, x1, y1) = w.rect();
        assert!(x1 >= x0 && y1 >= y0, "degenerate box for {:?}", w.text);
        assert!(x0 >= -5.0 && x1 <= 620.0, "x out of page range for {:?}", w.text);
        assert!(y0 >= -5.0 && y1 <= 800.0, "y out of page range for {:?}", w.text);
    }

    // word-count parity with the Python pipeline's golden _ocr.json (±25%:
    // pdfium and PyMuPDF may differ slightly on whitespace/token splitting)
    let golden = load_golden_ocr();
    let golden_count: usize = golden.iter().map(|p| p.len()).sum();
    let ratio = n as f64 / golden_count as f64;
    assert!(
        ratio > 0.75 && ratio < 1.33,
        "word count drifted from golden: rust={} golden={} ratio={:.2}",
        n,
        golden_count,
        ratio
    );
}

#[test]
fn annotates_with_golden_ocr_and_errors() {
    let pdf = bind();
    let golden_ocr = load_golden_ocr();
    let golden_errors: Vec<EatError> = serde_json::from_str(
        &std::fs::read_to_string(fixtures_dir().join("standard-test_errors.json")).unwrap(),
    )
    .unwrap();
    assert!(!golden_errors.is_empty(), "golden errors must be non-empty");

    let out = std::env::temp_dir().join("eat-rust-golden-annotated.pdf");
    let _ = std::fs::remove_file(&out);

    let ann = Annotator::new("en");
    let tags = ann
        .annotate_pdf(
            pdf.pdfium(),
            &fixtures_dir().join("standard-test.pdf").to_string_lossy(),
            &golden_ocr,
            &golden_errors,
            &out.to_string_lossy(),
        )
        .expect("annotate");

    assert!(tags > 0, "expected at least one sidebar tag from golden errors, got {}", tags);
    let meta = std::fs::metadata(&out).expect("annotated pdf written");
    assert!(meta.len() > 1000, "annotated pdf suspiciously small: {} bytes", meta.len());
}
