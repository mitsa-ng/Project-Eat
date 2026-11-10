//! eat-ocr — scanned/handwritten page OCR (Rust port of the RapidOCR path in
//! ocr_engine.py). Runs the exact same PP-OCRv6 ONNX models shipped with the
//! Python `rapidocr` package, via the `ort` ONNX Runtime bindings.
//!
//! Pipeline (mirrors rapidocr main.py):
//!   pre-resize (max side 2000) → det (DBNet) → sort boxes → perspective crop
//!   → cls (180° fix) → rec (CRNN + CTC) → filter text_score ≥ 0.5

pub mod geom;
pub mod imgops;

use geom::{get_mini_boxes, order_points_clockwise, point_in_polygon, Pt};
use imgops::{dilate2x2, Img};
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;

// ── tunables (rapidocr config.yaml defaults) ────────────────────────────────
const MAX_SIDE_LEN: usize = 2000;
const DET_LIMIT_SIDE_LEN: usize = 736; // limit_type = min
const DET_MEAN_STD: f32 = 0.5;
const DET_THRESH: f32 = 0.3;
const BOX_THRESH: f32 = 0.5;
const MAX_CANDIDATES: usize = 1000;
const UNCLIP_RATIO: f64 = 1.6;
const USE_DILATION: bool = true;
const MIN_SIZE: f64 = 3.0;
const BOX_SORT_Y_THRESHOLD: f64 = 10.0;

const TEXT_SCORE: f32 = 0.5;

const REC_FLIP_RETRY_THRESHOLD: f32 = 0.60;

const REC_IMAGE_H: usize = 48;
const REC_IMAGE_W: usize = 320; // base ratio; batch width grows dynamically
const REC_BATCH: usize = 6;

/// One recognised text line.
#[derive(Debug, Clone)]
pub struct OcrLine {
    /// 4 corner points in ORIGINAL image pixel coordinates (tl, tr, br, bl).
    pub box_: [(f64, f64); 4],
    pub text: String,
    pub score: f32,
}

#[derive(Debug)]
pub enum OcrError {
    ModelsNotFound(String),
    Ort(String),
    Dict(String),
}

impl std::fmt::Display for OcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcrError::ModelsNotFound(m) => write!(f, "PP-OCR models not found: {}", m),
            OcrError::Ort(m) => write!(f, "onnxruntime error: {}", m),
            OcrError::Dict(m) => write!(f, "rec dictionary error: {}", m),
        }
    }
}
impl std::error::Error for OcrError {}

fn ort_err(e: impl std::error::Error) -> OcrError {
    OcrError::Ort(e.to_string())
}

/// Python `round()` (banker's rounding) — rapidocr uses it for det resize.
fn py_round(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

pub struct RapidOcr {
    det: Session,
    rec: Session,
    /// CTC vocabulary: index 0 = blank, last = space.
    character: Vec<String>,
}

impl RapidOcr {
    /// Load the 3 ONNX models + dictionary. Resolution order for the model
    /// directory: `$EAT_PPROCR_DIR` → `models/` beside the executable →
    /// workspace `vendor/ppocr/` (dev layout).
    pub fn load() -> Result<Self, OcrError> {
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("EAT_PPROCR_DIR") {
            candidates.push(p.into());
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("models"));
                candidates.push(dir.join("ppocr"));
            }
        }
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            candidates.push(std::path::Path::new(&manifest).join("../../vendor/ppocr"));
        }
        let mut last = "no candidate dirs".to_string();
        for dir in &candidates {
            if dir.join("PP-OCRv6_det_small.onnx").exists() {
                return Self::load_from(dir);
            }
            last = format!("missing models in {}", dir.display());
        }
        Err(OcrError::ModelsNotFound(last))
    }

    pub fn load_from(dir: &std::path::Path) -> Result<Self, OcrError> {
        let env = ort::init().build().map_err(ort_err)?;
        let det = Session::builder(&env)
            .and_then(|mut b| b.commit_from_file(dir.join("PP-OCRv6_det_small.onnx")))
            .map_err(ort_err)?;
        let rec = Session::builder(&env)
            .and_then(|mut b| b.commit_from_file(dir.join("PP-OCRv6_rec_small.onnx")))
            .map_err(ort_err)?;

        let dict_raw = std::fs::read_to_string(dir.join("rec_keys.txt"))
            .map_err(|e| OcrError::Dict(format!("rec_keys.txt: {} ({})", e, dir.display())))?;
        let mut character: Vec<String> = dict_raw.split('\n').map(|s| s.to_string()).collect();
        // rapidocr CTCLabelDecode: append space, prepend blank
        character.push(" ".to_string());
        character.insert(0, "blank".to_string());

        Ok(RapidOcr { det, rec, character })
    }

    /// Run the full det→cls→rec pipeline on a BGR image. Returns lines in
    /// reading order (top→bottom, left→right within a line band).
    pub fn run(&mut self, img_bgr: &Img) -> Result<Vec<OcrLine>, OcrError> {
        if img_bgr.w < 2 || img_bgr.h < 2 {
            return Ok(Vec::new());
        }
        // global pre-resize: max side ≤ 2000 (track ratio to map boxes back)
        let scale = if img_bgr.w.max(img_bgr.h) > MAX_SIDE_LEN {
            MAX_SIDE_LEN as f64 / img_bgr.w.max(img_bgr.h) as f64
        } else {
            1.0
        };
        let img = if scale < 1.0 {
            img_bgr.resize(
                ((img_bgr.w as f64 * scale) as usize).max(1),
                ((img_bgr.h as f64 * scale) as usize).max(1),
            )
        } else {
            img_bgr.clone()
        };

        let boxes = self.detect(&img)?;
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        // sort in reading order
        let mut sorted = boxes;
        sorted.sort_by(|a, b| {
            let (ya, yb) = (a[0].1, b[0].1);
            let (line_a, line_b) = (ya / BOX_SORT_Y_THRESHOLD, yb / BOX_SORT_Y_THRESHOLD);
            line_a
                .partial_cmp(&line_b)
                .unwrap()
                .then(a[0].0.partial_cmp(&b[0].0).unwrap())
        });

        // perspective crops
        let crops: Vec<Img> = sorted
            .iter()
            .map(|b| {
                img.perspective_crop([
                    Pt::new(b[0].0, b[0].1),
                    Pt::new(b[1].0, b[1].1),
                    Pt::new(b[2].0, b[2].1),
                    Pt::new(b[3].0, b[3].1),
                ])
            })
            .collect();

        // Orientation handling: the upstream Python path uses a dedicated cls
        // model, whose behaviour regressed under ONNX Runtime ≥1.31 (it flips
        // horizontal handwriting upside-down). Instead we run rec once and
        // retry flipped only for low-confidence crops, keeping the better
        // score — robust across runtimes and still fixes upside-down scans.
        let (mut txts, mut scores) = self.recognize(&crops)?;
        for i in 0..crops.len() {
            if scores[i] < REC_FLIP_RETRY_THRESHOLD {
                let flipped = vec![crops[i].rotate180()];
                let (t2, s2) = self.recognize(&flipped)?;
                if s2[0] > scores[i] {
                    txts[i] = t2[0].clone();
                    scores[i] = s2[0];
                }
            }
        }
        if std::env::var("EAT_DEBUG_RAW").is_ok() {
            for (t, s) in txts.iter().zip(scores.iter()) {
                eprintln!("REC_RAW\t{:.3}\t{}", s, t);
            }
        }

        // assemble + filter (rapidocr text_score, empty txts); map boxes to
        // ORIGINAL image coordinates
        let inv = 1.0 / scale;
        let mut out = Vec::new();
        for (i, (txt, score)) in txts.iter().zip(scores.iter()).enumerate() {
            if txt.trim().is_empty() || *score < TEXT_SCORE {
                continue;
            }
            let b = sorted[i];
            out.push(OcrLine {
                box_: [
                    (b[0].0 * inv, b[0].1 * inv),
                    (b[1].0 * inv, b[1].1 * inv),
                    (b[2].0 * inv, b[2].1 * inv),
                    (b[3].0 * inv, b[3].1 * inv),
                ],
                text: txt.clone(),
                score: *score,
            });
        }
        Ok(out)
    }

    // ── detection ─────────────────────────────────────────────────────────────

    fn detect(&mut self, img: &Img) -> Result<Vec<[(f64, f64); 4]>, OcrError> {
        // det resize: multiple of 32, short side ≥ 736 (limit_type=min)
        let (h, w) = (img.h as f64, img.w as f64);
        let ratio = if w.min(h) < DET_LIMIT_SIDE_LEN as f64 {
            DET_LIMIT_SIDE_LEN as f64 / w.min(h)
        } else {
            1.0
        };
        let mut rw = py_round(py_round(w * ratio) / 32.0) * 32.0;
        let mut rh = py_round(py_round(h * ratio) / 32.0) * 32.0;
        rw = rw.max(32.0);
        rh = rh.max(32.0);
        let resized = img.resize(rw as usize, rh as usize);

        // normalize (x/255 - 0.5) / 0.5 → CHW
        let (rw, rh) = (rw as usize, rh as usize);
        let mut input = vec![0f32; 3 * rw * rh];
        for y in 0..rh {
            for x in 0..rw {
                let p = resized.px(x, y);
                let i = y * rw + x;
                input[i] = (p[0] as f32 / 255.0 - DET_MEAN_STD) / DET_MEAN_STD;
                input[rw * rh + i] = (p[1] as f32 / 255.0 - DET_MEAN_STD) / DET_MEAN_STD;
                input[2 * rw * rh + i] = (p[2] as f32 / 255.0 - DET_MEAN_STD) / DET_MEAN_STD;
            }
        }
        let tensor = Tensor::from_array(([1, 3, rh, rw], input)).map_err(ort_err)?;
        let outputs: SessionOutputs = self.det.run(ort::inputs!["x" => tensor]).map_err(ort_err)?;
        let arr = outputs["fetch_name_0"]
            .try_extract_array::<f32>()
            .map_err(ort_err)?;
        let shape = arr.shape().to_vec();
        let pred: Vec<f32> = arr.iter().copied().collect();
        if std::env::var("EAT_DUMP_PRED").is_ok() {
            let pw32 = shape[3] as u32;
            let ph32 = shape[2] as u32;
            let mut hdr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
                ph32, pw32
            )
            .into_bytes();
            while hdr.len() % 64 != 63 {
                hdr.push(b' ');
            }
            hdr.push(b'\n');
            let mut out: Vec<u8> = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
            out.extend_from_slice(&(hdr.len() as u16 + 1).to_le_bytes());
            // note: total header length includes padding already; simpler: rebuild canonical
            let mut out: Vec<u8> = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
            let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", ph32, pw32);
            while (h.len() + 10 + 1) % 64 != 0 {
                h.push(' ');
            }
            h.push('\n');
            out.extend_from_slice(&((h.len()) as u16).to_le_bytes());
            out.extend_from_slice(h.as_bytes());
            for v in &pred {
                out.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write("/tmp/rust-det-pred.npy", out).unwrap();
            eat_core::logging::info("det pred dumped → /tmp/rust-det-pred.npy");
        }
        // [1,1,H,W] → H×W plane
        let ph = shape[2] as usize;
        let pw = shape[3] as usize;
        debug_assert_eq!(pred.len(), ph * pw);

        Ok(db_postprocess(&pred, pw, ph, img.w, img.h))
    }

    // ── recognition ───────────────────────────────────────────────────────────

    fn recognize(&mut self, crops: &[Img]) -> Result<(Vec<String>, Vec<f32>), OcrError> {
        if crops.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        // sort by aspect ratio (w/h) ascending for batching efficiency
        let mut indices: Vec<usize> = (0..crops.len()).collect();
        indices.sort_by(|&a, &b| {
            let ra = crops[a].w as f32 / crops[a].h as f32;
            let rb = crops[b].w as f32 / crops[b].h as f32;
            ra.partial_cmp(&rb).unwrap()
        });

        let mut texts: Vec<String> = vec![String::new(); crops.len()];
        let mut scores: Vec<f32> = vec![0.0; crops.len()];

        for beg in (0..indices.len()).step_by(REC_BATCH) {
            let batch_idx = &indices[beg..(beg + REC_BATCH).min(indices.len())];
            // per-batch dynamic width
            let mut max_ratio = REC_IMAGE_W as f32 / REC_IMAGE_H as f32; // 6.667
            for &i in batch_idx {
                let r = crops[i].w as f32 / crops[i].h as f32;
                max_ratio = max_ratio.max(r);
            }
            let img_w = (REC_IMAGE_H as f32 * max_ratio) as usize;
            let mut batch: Vec<f32> = Vec::with_capacity(batch_idx.len() * 3 * REC_IMAGE_H * img_w);
            for &i in batch_idx {
                batch.extend_from_slice(&rec_norm(&crops[i], img_w));
            }
            let tensor = Tensor::from_array(([batch_idx.len(), 3, REC_IMAGE_H, img_w], batch))
                .map_err(ort_err)?;
            let outputs = self.rec.run(ort::inputs!["x" => tensor]).map_err(ort_err)?;
            let preds_arr = outputs["fetch_name_0"]
                .try_extract_array::<f32>()
                .map_err(ort_err)?;
            let shape = preds_arr.shape().to_vec();
            let preds: Vec<f32> = preds_arr.iter().copied().collect();
            let n = shape[0] as usize;
            let t = shape[1] as usize;
            let c = shape[2] as usize;
            debug_assert_eq!(c, self.character.len());

            for row in 0..n {
                let (txt, score) = ctc_decode(&preds[row * t * c..(row + 1) * t * c], t, c, &self.character);
                let orig = batch_idx[row];
                texts[orig] = txt;
                scores[orig] = score;
            }
        }
        Ok((texts, scores))
    }
}

// ── normalization helpers ─────────────────────────────────────────────────────

/// rec resize_norm: h=48 keep-ratio (cap w=batch_w), normalize, zero-pad.
fn rec_norm(img: &Img, batch_w: usize) -> Vec<f32> {
    let ratio = img.w as f32 / img.h as f32;
    let resized_w = ((((REC_IMAGE_H as f32 * ratio).ceil() as usize).min(batch_w)).max(1)).min(batch_w);
    let resized = img.resize(resized_w, REC_IMAGE_H);
    let mut out = vec![0f32; 3 * REC_IMAGE_H * batch_w];
    for y in 0..REC_IMAGE_H {
        for x in 0..resized_w {
            let p = resized.px(x, y);
            let i = y * batch_w + x;
            out[i] = (p[0] as f32 / 255.0 - 0.5) / 0.5;
            out[REC_IMAGE_H * batch_w + i] = (p[1] as f32 / 255.0 - 0.5) / 0.5;
            out[2 * REC_IMAGE_H * batch_w + i] = (p[2] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    out
}

/// Greedy CTC decode: argmax per step, collapse repeats, drop blank (index 0).
/// Confidence = mean of the kept steps' probabilities (rounded to 5 dp).
fn ctc_decode(preds: &[f32], t: usize, c: usize, character: &[String]) -> (String, f32) {
    let mut text = String::new();
    let mut kept_probs: Vec<f32> = Vec::new();
    let mut prev_idx: isize = -1;
    for step in 0..t {
        let row = &preds[step * c..(step + 1) * c];
        let (mut best_i, mut best_p) = (0usize, f32::MIN);
        for (i, &p) in row.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_i = i;
            }
        }
        if best_i as isize == prev_idx {
            continue; // collapse repeats
        }
        prev_idx = best_i as isize;
        if best_i == 0 {
            continue; // blank
        }
        text.push_str(&character[best_i]);
        kept_probs.push((best_p * 100000.0).round() / 100000.0);
    }
    let score = if kept_probs.is_empty() {
        0.0
    } else {
        (kept_probs.iter().sum::<f32>() / kept_probs.len() as f32 * 100000.0).round() / 100000.0
    };
    (text, score)
}

/// Port of DBPostProcess.box_score_fast: mean det probability inside the
/// polygon (rasterized mask over the AABB).
fn box_score_fast(pred: &[f32], w: usize, h: usize, poly: &[Pt]) -> f32 {
    let xmin = poly.iter().map(|p| p.x.floor()).fold(f64::INFINITY, f64::min).clamp(0.0, (w - 1) as f64) as usize;
    let xmax = poly.iter().map(|p| p.x.ceil()).fold(f64::NEG_INFINITY, f64::max).clamp(0.0, (w - 1) as f64) as usize;
    let ymin = poly.iter().map(|p| p.y.floor()).fold(f64::INFINITY, f64::min).clamp(0.0, (h - 1) as f64) as usize;
    let ymax = poly.iter().map(|p| p.y.ceil()).fold(f64::NEG_INFINITY, f64::max).clamp(0.0, (h - 1) as f64) as usize;
    let mut sum = 0f32;
    let mut count = 0u32;
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            if point_in_polygon(x as f64 + 0.5, y as f64 + 0.5, poly) {
                sum += pred[y * w + x];
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// 8-connected component labeling (iterative flood fill).
fn connected_components(mask: &[u8], w: usize, h: usize) -> Vec<Vec<(usize, usize)>> {
    let mut visited = vec![false; mask.len()];
    let mut comps = Vec::new();
    for start in 0..mask.len() {
        if visited[start] || mask[start] == 0 {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            comp.push((x, y));
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = (ny as usize) * w + nx as usize;
                    if !visited[j] && mask[j] != 0 {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        comps.push(comp);
    }
    comps
}

/// DB post-process (port of rapidocr DBPostProcess with the v6 config):
/// threshold → dilate → components → min-area rect → score → unclip → scale.
pub fn db_postprocess(
    pred: &[f32],
    pred_w: usize,
    pred_h: usize,
    img_w: usize,
    img_h: usize,
) -> Vec<[(f64, f64); 4]> {
    let bitmap: Vec<u8> = pred.iter().map(|&p| if p > DET_THRESH { 1 } else { 0 }).collect();
    let mask = if USE_DILATION {
        dilate2x2(&bitmap, pred_w, pred_h)
    } else {
        bitmap
    };

    let comps = connected_components(&mask, pred_w, pred_h);

    let mut boxes: Vec<[(f64, f64); 4]> = Vec::new();
    let mut taken = 0usize;
    for comp in comps {
        if taken >= MAX_CANDIDATES {
            break;
        }
        taken += 1;
        if comp.len() < 3 {
            continue;
        }
        let pts: Vec<Pt> = comp.iter().map(|&(x, y)| Pt::new(x as f64, y as f64)).collect();
        let (mini, sside) = get_mini_boxes(&pts);
        if sside < MIN_SIZE {
            continue;
        }
        let score = box_score_fast(pred, pred_w, pred_h, &mini);
        if score < BOX_THRESH {
            continue;
        }
        let expanded = geom::unclip_rect(&mini, UNCLIP_RATIO);
        let (box_pts, sside2) = get_mini_boxes(&expanded);
        if sside2 < MIN_SIZE + 2.0 {
            continue;
        }
        let sx = img_w as f64 / pred_w as f64;
        let sy = img_h as f64 / pred_h as f64;
        let scaled: Vec<Pt> = box_pts.iter().map(|p| Pt::new(p.x * sx, p.y * sy)).collect();
        let mut ordered = order_points_clockwise([scaled[0], scaled[1], scaled[2], scaled[3]]);
        for p in ordered.iter_mut() {
            p.x = p.x.clamp(0.0, (img_w - 1) as f64).round();
            p.y = p.y.clamp(0.0, (img_h - 1) as f64).round();
        }
        let w_len = ordered[1].sub(ordered[0]).norm();
        let h_len = ordered[0].sub(ordered[3]).norm();
        if w_len <= 3.0 || h_len <= 3.0 {
            continue;
        }
        boxes.push([
            (ordered[0].x, ordered[0].y),
            (ordered[1].x, ordered[1].y),
            (ordered[2].x, ordered[2].y),
            (ordered[3].x, ordered[3].y),
        ]);
    }
    boxes
}

impl RapidOcr {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_area_rect_axis_aligned() {
        let pts: Vec<Pt> = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
            .iter()
            .map(|&(x, y)| Pt::new(x, y))
            .collect();
        let (corners, w, h) = geom::min_area_rect(&pts);
        assert!((w - 10.0).abs() < 1e-6 && (h - 4.0).abs() < 1e-6, "w={} h={}", w, h);
        assert_eq!(corners.len(), 4);
    }

    #[test]
    fn min_area_rect_rotated() {
        // a thin 45° line: min-area rect is degenerate (w=L√2? no — w≈L*1 along, h≈0)
        let pts: Vec<Pt> = [(0.0, 0.0), (10.0, 10.0), (5.0, 5.0)]
            .iter()
            .map(|&(x, y)| Pt::new(x, y))
            .collect();
        let (_, w, h) = geom::min_area_rect(&pts);
        // diagonal length ~14.14, perpendicular extent ~0
        assert!(w.max(h) > 14.0 && w.min(h) < 0.1, "w={} h={}", w, h);
    }

    #[test]
    fn unclip_grows_rect_by_expected_dist() {
        // 100×20 rect: area=2000, perimeter=240 → dist = 2000*1.6/240 ≈ 13.33
        let pts: Vec<Pt> = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)]
            .iter()
            .map(|&(x, y)| Pt::new(x, y))
            .collect();
        let out = geom::unclip_rect(&pts, 1.6);
        let w = out[1].sub(out[0]).norm();
        let h = out[0].sub(out[3]).norm();
        assert!((w - (100.0 + 2.0 * 13.333)).abs() < 0.5, "w={}", w);
        assert!((h - (20.0 + 2.0 * 13.333)).abs() < 0.5, "h={}", h);
    }

    #[test]
    fn ctc_decode_collapses_and_drops_blank() {
        // vocab: blank, a, b, space
        let character = vec!["blank".into(), "a".into(), "b".into(), " ".into()];
        // steps: a a blank b b a → "a" then "b" then "a" = "aba"
        let mut preds = vec![0f32; 6 * 4];
        let set = |p: &mut Vec<f32>, step: usize, idx: usize| {
            for c in 0..4 {
                p[step * 4 + c] = 0.1;
            }
            p[step * 4 + idx] = 0.9;
        };
        set(&mut preds, 0, 1); // a
        set(&mut preds, 1, 1); // a (repeat)
        set(&mut preds, 2, 0); // blank
        set(&mut preds, 3, 2); // b
        set(&mut preds, 4, 2); // b (repeat)
        set(&mut preds, 5, 1); // a (new after b)
        let (text, score) = ctc_decode(&preds, 6, 4, &character);
        assert_eq!(text, "aba");
        assert!((score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn py_round_bankers() {
        assert_eq!(py_round(62.5), 62.0);
        assert_eq!(py_round(63.5), 64.0);
        assert_eq!(py_round(44.15625), 44.0);
        assert_eq!(py_round(0.5), 0.0);
    }

    #[test]
    fn dilate_grows_one_pixel_right_down() {
        // single pixel at (2,2) in 5×5 → covers (2,2),(3,2),(2,3),(3,3)
        let mut mask = vec![0u8; 25];
        mask[2 * 5 + 2] = 1;
        let out = dilate2x2(&mask, 5, 5);
        assert_eq!(&out[2 * 5 + 2..2 * 5 + 4], &[1, 1]);
        assert_eq!(&out[3 * 5 + 2..3 * 5 + 4], &[1, 1]);
        assert_eq!(out[1 * 5 + 1], 0);
        assert_eq!(out.iter().sum::<u8>(), 4);
    }
}
