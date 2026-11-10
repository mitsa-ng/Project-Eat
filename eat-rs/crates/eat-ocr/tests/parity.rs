//! Parity tests: Rust eat-ocr vs the Python RapidOCR golden capture
//! (same PP-OCRv6 ONNX models, same 300-DPI render of the handwritten test).

use eat_ocr::imgops::Img;
use eat_ocr::{RapidOcr, OcrError};

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .unwrap()
}

fn load_ocr() -> Result<RapidOcr, OcrError> {
    let dir = fixtures().join("../../vendor/ppocr");
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    std::env::set_var("EAT_PPROCR_DIR", &dir);
    RapidOcr::load()
}

fn load_rgb(path: &std::path::Path) -> Img {
    let img = image::open(path).expect("decode png");
    let rgb = img.to_rgb8();
    Img::from_rgb(rgb.width() as usize, rgb.height() as usize, &rgb)
}

#[derive(serde::Deserialize)]
struct GoldenLine {
    pub box_: Vec<Vec<f64>>, // serde rename below via alias
    pub text: String,
    pub score: f32,
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn similarity(a: &str, b: &str) -> f64 {
    let max = a.chars().count().max(b.chars().count()).max(1);
    1.0 - levenshtein(a, b) as f64 / max as f64
}

fn poly_iou(a: &[(f64, f64); 4], b: &Vec<Vec<f64>>) -> f64 {
    // AABB IoU is sufficient for positional parity checking
    let rect = |pts: &[(f64, f64)]| {
        let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
        (
            xs.iter().cloned().fold(f64::INFINITY, f64::min),
            ys.iter().cloned().fold(f64::INFINITY, f64::min),
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let a2: Vec<(f64, f64)> = b.iter().map(|p| (p[0], p[1])).collect();
    let (ax0, ay0, ax1, ay1) = rect(a);
    let (bx0, by0, bx1, by1) = rect(&a2);
    let ix = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let iy = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = ix * iy;
    let union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[test]
fn handwritten_parity_vs_python_rapidocr() {
    let mut ocr = load_ocr().expect("load models");

    // golden (serde: Python used "box" as the key)
    #[derive(serde::Deserialize)]
    struct GoldenLineRaw {
        r#box: Vec<Vec<f64>>,
        text: String,
        score: f32,
    }
    let golden_raw: Vec<GoldenLineRaw> = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("hw-golden.json")).unwrap(),
    )
    .unwrap();
    let golden: Vec<GoldenLine> = golden_raw
        .into_iter()
        .map(|g| GoldenLine {
            box_: g.r#box,
            text: g.text,
            score: g.score,
        })
        .collect();
    let n_golden = golden.len();
    assert!(n_golden >= 20, "golden suspiciously small: {}", n_golden);

    let img = load_rgb(&fixtures().join("hw-test-p1.png"));
    let t0 = std::time::Instant::now();
    let lines = ocr.run(&img).expect("ocr run");
    eprintln!(
        "rust ocr: {} lines in {:.1}s (python golden: {})",
        lines.len(),
        t0.elapsed().as_secs_f32(),
        n_golden
    );

    // 1. line count within ±20%
    let ratio = lines.len() as f64 / n_golden as f64;
    assert!(
        ratio > 0.8 && ratio < 1.2,
        "line count drifted: rust={} golden={} ratio={:.2}",
        lines.len(),
        n_golden,
        ratio
    );

    // 2. text match: every golden line should find a Rust counterpart with
    //    similarity ≥ 0.8 (exact text expected for most; CTC + identical
    //    models should agree closely)
    let mut exact = 0usize;
    let mut fuzzy = 0usize;
    let mut missed: Vec<&str> = Vec::new();
    for g in &golden {
        let mut best = 0.0;
        for l in &lines {
            let s = similarity(&g.text, &l.text);
            if s > best {
                best = s;
            }
        }
        if best >= 0.999 {
            exact += 1;
        } else if best >= 0.8 {
            fuzzy += 1;
        } else {
            missed.push(&g.text);
        }
    }
    let matched = exact + fuzzy;
    eprintln!(
        "text parity: {} exact, {} fuzzy, {} missed (of {})",
        exact,
        fuzzy,
        missed.len(),
        n_golden
    );
    assert!(
        matched as f64 / n_golden as f64 >= 0.75,
        "text parity too low: {}/{} matched; missed examples: {:?}",
        matched,
        n_golden,
        &missed[..missed.len().min(5)]
    );

    // 3. positional parity: matched-by-text pairs must overlap (IoU ≥ 0.5)
    let mut ious = Vec::new();
    for g in &golden {
        let mut best: Option<(f64, &eat_ocr::OcrLine)> = None;
        for l in &lines {
            let s = similarity(&g.text, &l.text);
            if s >= 0.8 && best.map(|(bs, _)| s > bs).unwrap_or(true) {
                best = Some((s, l));
            }
        }
        if let Some((_, l)) = best {
            ious.push(poly_iou(&l.box_, &g.box_));
        }
    }
    let avg_iou = ious.iter().sum::<f64>() / ious.len().max(1) as f64;
    eprintln!("positional parity: avg IoU {:.3} over {} pairs", avg_iou, ious.len());
    assert!(avg_iou >= 0.5, "avg IoU too low: {:.3}", avg_iou);
}
