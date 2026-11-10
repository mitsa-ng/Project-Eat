//! eat-pdf — PDF layer (Rust port of ocr_engine.py's digital path + annotator.py).
//!
//! Uses pdfium (Chrome's PDF engine) where the Python pipeline used PyMuPDF.
//! Coordinate convention matches the Python pipeline's JSON schema: all word
//! boxes are in PDF point-space with origin at the TOP-LEFT of the page;
//! pdfium's native bottom-up coordinates are flipped at this boundary.

use eat_core::languages::{is_cjk, normalize};
use eat_core::nlp::EatError;
use eat_core::words::{Pages, Word};
use pdfium_render::prelude::*;
use std::path::{Path, PathBuf};

// ── tunables (ported from ocr_engine.py / annotator.py) ────────────────────
pub const DIGITAL_THRESHOLD: usize = 15;

/// Per-page extraction result: the words found and how they were found.
#[derive(Debug)]
pub struct PageExtract {
    pub words: Vec<Word>,
    /// true when the page fell below DIGITAL_THRESHOLD native words and would
    /// have gone to RapidOCR in the Python pipeline. (Scanned-page OCR is
    /// phase 2 of the Rust port; such pages currently yield no words.)
    pub scanned: bool,
}

#[derive(Debug)]
pub enum PdfError {
    Bind(String),
    Load(String),
    Save(String),
    Internal(String),
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdfError::Bind(m) => write!(f, "pdfium bind failed: {}", m),
            PdfError::Load(m) => write!(f, "PDF load failed: {}", m),
            PdfError::Save(m) => write!(f, "PDF save failed: {}", m),
            PdfError::Internal(m) => write!(f, "{}", m),
        }
    }
}
impl std::error::Error for PdfError {}

fn internal(e: PdfiumError) -> PdfError {
    PdfError::Internal(e.to_string())
}

/// One bound pdfium instance per process (pdfium-render initializes bindings
/// exactly once). Cheap to reuse for extraction and annotation.
pub struct EatPdf {
    pdfium: Pdfium,
}

impl EatPdf {
    /// Bind to the pdfium dynamic library, trying, in order:
    /// 1. `$EAT_PDFIUM_LIB` (absolute path to libpdfium)
    /// 2. `libpdfium` next to the running executable (installed-app layout)
    /// 3. `vendor/pdfium-macos-arm64/lib/libpdfium.dylib` under the workspace
    pub fn bind() -> Result<Self, PdfError> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("EAT_PDFIUM_LIB") {
            candidates.push(PathBuf::from(p));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join(pdfium_platform_dylib_name()));
                candidates.push(dir.join("lib").join(pdfium_platform_dylib_name()));
            }
        }
        // dev workspace layout: eat-rs/vendor/pdfium-macos-arm64/lib/libpdfium.dylib
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            candidates.push(
                Path::new(&manifest)
                    .join("../../vendor/pdfium-macos-arm64/lib")
                    .join(pdfium_platform_dylib_name()),
            );
        }
        let mut last = String::from("no candidate paths");
        for c in &candidates {
            if c.exists() {
                match Pdfium::bind_to_library(c) {
                    Ok(b) => return Ok(EatPdf { pdfium: Pdfium::new(b) }),
                    Err(e) => last = format!("{} ({})", e, c.display()),
                }
            } else {
                last = format!("not found: {}", c.display());
            }
        }
        Err(PdfError::Bind(last))
    }

    // ── digital word extraction (port of OCREngine._extract_digital) ────────

    fn extract_page_words(page: &PdfPage) -> Result<Vec<Word>, PdfError> {
        let page_h = page.height().value;
        let mut words: Vec<Word> = Vec::new();
        let mut text = String::new();
        let mut x0 = f32::MAX;
        let mut y0 = f32::MAX;
        let mut x1 = f32::MIN;
        let mut y1 = f32::MIN;

        fn flush(
            text: &mut String,
            x0: &mut f32,
            y0: &mut f32,
            x1: &mut f32,
            y1: &mut f32,
            words: &mut Vec<Word>,
            page_h: f32,
        ) {
            let t = text.trim().to_string();
            if !t.is_empty() && *x1 > *x0 {
                // top-left-origin box: flip pdfium's bottom-up Y
                let top = page_h - *y1;
                let bottom = page_h - *y0;
                words.push(Word {
                    text: t,
                    box_: [
                        [*x0 as f64, top as f64],
                        [*x1 as f64, top as f64],
                        [*x1 as f64, bottom as f64],
                        [*x0 as f64, bottom as f64],
                    ],
                });
            }
            text.clear();
            *x0 = f32::MAX;
            *y0 = f32::MAX;
            *x1 = f32::MIN;
            *y1 = f32::MIN;
        }

        let text_page = page.text().map_err(internal)?;
        let all_chars = text_page.chars();
        for ch in all_chars.iter() {
            match ch.unicode_char() {
                Some(c) if !c.is_whitespace() => {
                    if let Ok(b) = ch.tight_bounds() {
                        x0 = x0.min(b.left().value);
                        x1 = x1.max(b.right().value);
                        y0 = y0.min(b.bottom().value);
                        y1 = y1.max(b.top().value);
                    }
                    text.push(c);
                }
                _ => flush(&mut text, &mut x0, &mut y0, &mut x1, &mut y1, &mut words, page_h),
            }
        }
        flush(&mut text, &mut x0, &mut y0, &mut x1, &mut y1, &mut words, page_h);
        Ok(words)
    }

    /// Port of OCREngine.process_page's dispatch: native extraction, digital
    /// threshold + low-quality heuristic. Scanned pages return `scanned: true`
    /// with zero words (RapidOCR path is phase 2 of the Rust port).
    pub fn extract_pdf(&self, path: &str) -> Result<Vec<PageExtract>, PdfError> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .map_err(|e| PdfError::Load(format!("{} ({})", e, path)))?;
        let count = doc.pages().len();
        let mut out = Vec::new();
        for i in 0..count {
            let page = doc.pages().get(i).map_err(internal)?;
            let words = Self::extract_page_words(&page)?;
            let n = words.len();
            let mut scanned = false;
            if n >= DIGITAL_THRESHOLD {
                // guard: pages with furniture only (page numbers, headers)
                if n < DIGITAL_THRESHOLD * 2 {
                    let short = words
                        .iter()
                        .filter(|w| {
                            w.text.chars().count() <= 3
                                || !w.text.is_empty() && w.text.chars().all(|c| c.is_ascii_digit())
                        })
                        .count();
                    let ratio = short as f64 / n as f64;
                    if ratio > 0.6 {
                        scanned = true;
                    }
                }
            } else {
                scanned = true;
            }
            out.push(PageExtract { words, scanned });
        }
        Ok(out)
    }

    pub fn pdfium(&self) -> &Pdfium {
        &self.pdfium
    }

    /// Render a page to RGB bytes at the given DPI (for the scanned-page OCR
    /// path — the Rust equivalent of ocr_engine's `page.get_pixmap` step).
    pub fn render_page_rgb(
        &self,
        path: &str,
        page_idx: i32,
        dpi: f32,
    ) -> Result<(u32, u32, Vec<u8>), PdfError> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .map_err(|e| PdfError::Load(format!("{} ({})", e, path)))?;
        let page = doc.pages().get(page_idx).map_err(internal)?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);
        let bitmap = page.render_with_config(&config).map_err(internal)?;
        let img = bitmap.as_image().map_err(internal)?.to_rgb8();
        Ok((img.width(), img.height(), img.into_raw()))
    }
}

// ── annotator (port of annotator.py — sidebar tags) ─────────────────────────

// visual constants (identical values to annotator.py)
const HIGHLIGHT_OPACITY: f32 = 0.22;
const UNDERLINE_WIDTH: f32 = 1.2;
const BORDER_WIDTH: f32 = 0.5;
const SIDEBAR_W: f32 = 72.0;
const SIDEBAR_GUTTER: f32 = 4.0;
const SIDEBAR_PAD_TOP: f32 = 4.0;
const TAG_H: f32 = 9.0;
const TAG_GAP: f32 = 2.0;
const TAG_FONT_SIZE: f32 = 6.0;
const TAG_PAD_X: f32 = 3.0;
const LEGEND_W: f32 = 62.0;
const LEGEND_H: f32 = 9.0;
const LEGEND_GAP: f32 = 2.0;
const LEGEND_MARGIN_T: f32 = 4.0;

const LEGEND_ITEMS: [(&str, (u8, u8, u8)); 3] = [
    ("Spelling", (255, 68, 68)),
    ("Grammar", (255, 140, 0)),
    ("Semantic", (30, 144, 255)),
];

fn color_for(kind: &str) -> (u8, u8, u8) {
    match kind {
        "spelling" => (255, 68, 68),   // (1.0, 0.267, 0.267)
        "semantic" => (30, 144, 255),  // (0.118, 0.565, 1.0)
        _ => (255, 140, 0),            // grammar (1.0, 0.549, 0.0)
    }
}

/// Axis-aligned rect in TOP-LEFT origin coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RectTd {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl RectTd {
    fn from_word(w: &Word) -> Self {
        let (x0, y0, x1, y1) = w.rect();
        RectTd { x0: x0 as f32, y0: y0 as f32, x1: x1 as f32, y1: y1 as f32 }
    }
    fn union(&self, o: &RectTd) -> RectTd {
        RectTd {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }
    fn center_y(&self) -> f32 {
        (self.y0 + self.y1) / 2.0
    }
}

/// Python `re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()` — \w is unicode-aware.
fn clean_token(token: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    token.trim_matches(|c: char| !is_word(c)).to_lowercase()
}

/// Port of Annotator._group_by_line: merge rects whose vertical centres are
/// within `tol` points; returns one merged rect per line segment.
fn group_by_line(rects: &[RectTd], tol: f32) -> Vec<RectTd> {
    if rects.is_empty() {
        return Vec::new();
    }
    let mut groups: Vec<RectTd> = Vec::new();
    let mut cur: Vec<RectTd> = vec![rects[0]];
    for r in &rects[1..] {
        let prev_cy = cur.last().unwrap().center_y();
        if (r.center_y() - prev_cy).abs() <= tol {
            cur.push(*r);
        } else {
            let merged = cur.iter().skip(1).fold(cur[0], |acc, x| acc.union(x));
            groups.push(merged);
            cur = vec![*r];
        }
    }
    let merged = cur.iter().skip(1).fold(cur[0], |acc, x| acc.union(x));
    groups.push(merged);
    groups
}

/// Port of Annotator._find_spans_ocr: one rect per line segment of each match.
fn find_spans_ocr(target: &str, page_words: &[Word]) -> Vec<RectTd> {
    let target_tokens: Vec<String> = target
        .split_whitespace()
        .filter(|t| !t.trim().is_empty())
        .map(clean_token)
        .filter(|t| !t.is_empty())
        .collect();
    if target_tokens.is_empty() {
        return Vec::new();
    }
    let n = target_tokens.len();
    let cleaned: Vec<String> = page_words.iter().map(|w| clean_token(&w.text)).collect();
    let mut all_rects = Vec::new();
    let mut i = 0usize;
    while i + n <= cleaned.len() {
        if cleaned[i..i + n] == target_tokens[..] {
            let word_rects: Vec<RectTd> =
                (0..n).map(|j| RectTd::from_word(&page_words[i + j])).collect();
            all_rects.extend(group_by_line(&word_rects, 4.0));
            i += n;
        } else {
            i += 1;
        }
    }
    all_rects
}

pub struct Annotator {
    pub language: String,
    cjk: bool,
}

impl Annotator {
    pub fn new(language: &str) -> Self {
        let language = normalize(language).to_string();
        let cjk = is_cjk(&language);
        if cjk {
            // Helvetica (WinAnsi) cannot render CJK tag text; CJK-capable font
            // embedding is phase 2 of the Rust port. Tags still get correct
            // geometry and colour; suggestion glyphs may not render.
            eat_core::logging::warn(
                "  CJK tag font not yet bundled; sidebar suggestions may not render glyphs",
            );
        }
        Annotator { language, cjk }
    }

    fn alpha(c: (u8, u8, u8), opacity: f32) -> PdfColor {
        PdfColor::new(c.0, c.1, c.2, (opacity * 255.0).round() as u8)
    }

    /// Annotate `input_pdf` with `errors` and save to `output_pdf`.
    /// Returns the number of sidebar tags drawn. Port of Annotator.annotate_pdf.
    pub fn annotate_pdf(
        &self,
        pdfium: &Pdfium,
        input_pdf: &str,
        pages: &Pages,
        errors: &[EatError],
        output_pdf: &str,
    ) -> Result<usize, PdfError> {
        let mut doc = pdfium
            .load_pdf_from_file(input_pdf, None)
            .map_err(|e| PdfError::Load(format!("{} ({})", e, input_pdf)))?;
        let font = doc.fonts_mut().helvetica();
        let mut total = 0usize;
        let count = doc.pages().len();

        for page_num in 0..count {
            let mut page = doc.pages().get(page_num).map_err(internal)?;
            eat_core::logging::info(&format!(
                "Annotating page {}/{} …",
                page_num + 1,
                count
            ));

            // page-tagged errors only match on their own page (port behaviour)
            let page_errors: Vec<&EatError> = errors
                .iter()
                .filter(|e| e.page.is_none() || e.page == Some(page_num as u32))
                .collect();
            let empty_words: Vec<Word> = Vec::new();
            let page_words: &Vec<Word> = pages
                .get(page_num as usize)
                .unwrap_or(&empty_words);

            total += self.annotate_page(&mut page, &page_words, &page_errors, &font)?;
        }

        doc.save_to_file(output_pdf)
            .map_err(|e| PdfError::Save(format!("{} ({})", e, output_pdf)))?;
        eat_core::logging::info(&format!("Saved → {}  ({} annotation(s))", output_pdf, total));
        Ok(total)
    }

    fn annotate_page(
        &self,
        page: &mut PdfPage,
        page_words: &[Word],
        errors: &[&EatError],
        font: &PdfFontToken,
    ) -> Result<usize, PdfError> {
        let pw = page.width().value;
        let ph = page.height().value;
        let fy = |y_td: f32| ph - y_td;
        let sidebar_x0 = pw - SIDEBAR_W;
        let sidebar_x1 = pw - 2.0;

        // sidebar divider (full-height vertical line)
        let sx0_td = pw - SIDEBAR_W - SIDEBAR_GUTTER / 2.0;
        page.objects_mut()
            .create_path_object_line(
                PdfPoints::new(sx0_td),
                PdfPoints::new(fy(0.0)),
                PdfPoints::new(sx0_td),
                PdfPoints::new(fy(ph)),
                PdfColor::new(217, 217, 217, 255),
                PdfPoints::new(0.4),
            )
            .map_err(internal)?;

        // legend (colour swatch + label per error type)
        let legend_x0 = (pw - SIDEBAR_W) + (SIDEBAR_W - LEGEND_W) / 2.0;
        let legend_x1 = legend_x0 + LEGEND_W;
        for (i, &(label, c)) in LEGEND_ITEMS.iter().enumerate() {
            let y0 = LEGEND_MARGIN_T + i as f32 * (LEGEND_H + LEGEND_GAP);
            page.objects_mut()
                .create_path_object_rect(
                    PdfRect::new_from_values(fy(y0 + LEGEND_H), legend_x0, fy(y0), legend_x1),
                    None,
                    None,
                    Some(Self::alpha(c, 0.80)),
                )
                .map_err(internal)?;
            // Label without the Python version's "■" prefix: the swatch rect
            // already carries the colour and Helvetica has no U+25A0 glyph.
            page.objects_mut()
                .create_text_object(
                    PdfPoints::new(legend_x0 + 4.0),
                    PdfPoints::new(fy(y0 + LEGEND_H - 3.0)),
                    label,
                    font.clone(),
                    PdfPoints::new(TAG_FONT_SIZE),
                )
                .map_err(internal)?
                .set_fill_color(PdfColor::new(255, 255, 255, 255))
                .map_err(internal)?;
        }

        let legend_bottom = LEGEND_MARGIN_T + LEGEND_ITEMS.len() as f32 * (LEGEND_H + LEGEND_GAP) + 6.0;
        let mut next_tag_y = legend_bottom.max(SIDEBAR_PAD_TOP);

        // Step 1: collect (span_rect, kind, color, label)
        let mut pairs: Vec<(RectTd, String, (u8, u8, u8), String)> = Vec::new();
        let mut seen_pos: std::collections::HashSet<(i32, i32, i32, i32)> =
            std::collections::HashSet::new();

        for error in errors {
            let kind = if error.kind.is_empty() { "grammar" } else { error.kind.as_str() };
            let color = color_for(kind);
            let spans = find_spans_ocr(&error.original, page_words);
            if spans.is_empty() {
                eat_core::logging::debug(&format!("  No match: {:?}", error.original));
                continue;
            }
            let char_factor = if self.cjk { 0.95 } else { 0.52 };
            let max_chars = ((SIDEBAR_W - TAG_PAD_X * 2.0) / (TAG_FONT_SIZE * char_factor)) as usize;
            let keep = max_chars.saturating_sub(6);
            let suggestion_short: String = error.suggestion.chars().take(keep).collect();
            let tag_kind: String = kind
                .chars()
                .take(3)
                .flat_map(char::to_uppercase)
                .collect();
            let label = format!("[{}] {}", tag_kind, suggestion_short);

            for span in spans {
                let key = (
                    span.x0.round() as i32,
                    span.y0.round() as i32,
                    span.x1.round() as i32,
                    span.y1.round() as i32,
                );
                if !seen_pos.insert(key) {
                    continue;
                }
                pairs.push((span, kind.to_string(), color, label.clone()));
            }
        }
        if pairs.is_empty() {
            return Ok(0);
        }

        // Step 2: sort top → bottom
        pairs.sort_by(|a, b| a.0.center_y().partial_cmp(&b.0.center_y()).unwrap());

        // Step 3: highlights + underlines
        for (span, _, color, _) in &pairs {
            page.objects_mut()
                .create_path_object_rect(
                    PdfRect::new_from_values(fy(span.y1), span.x0, fy(span.y0), span.x1),
                    Some(PdfColor::new(color.0, color.1, color.2, 255)),
                    Some(PdfPoints::new(BORDER_WIDTH)),
                    Some(Self::alpha(*color, HIGHLIGHT_OPACITY)),
                )
                .map_err(internal)?;
            let uy = span.y1 + 1.0;
            page.objects_mut()
                .create_path_object_line(
                    PdfPoints::new(span.x0),
                    PdfPoints::new(fy(uy)),
                    PdfPoints::new(span.x1),
                    PdfPoints::new(fy(uy)),
                    PdfColor::new(color.0, color.1, color.2, 255),
                    PdfPoints::new(UNDERLINE_WIDTH),
                )
                .map_err(internal)?;
        }

        // Step 4: sidebar tags (packed top-to-bottom, collision-avoided)
        let mut count = 0usize;
        for (span, _, color, label) in &pairs {
            let ideal_y0 = span.center_y() - TAG_H / 2.0;
            let mut tag_y0 = ideal_y0.max(next_tag_y);
            let mut tag_y1 = tag_y0 + TAG_H;
            if tag_y1 > ph - 2.0 {
                tag_y1 = ph - 2.0;
                tag_y0 = tag_y1 - TAG_H;
            }
            page.objects_mut()
                .create_path_object_rect(
                    PdfRect::new_from_values(fy(tag_y1), sidebar_x0 + 2.0, fy(tag_y0), sidebar_x1),
                    None,
                    None,
                    Some(Self::alpha(*color, 0.90)),
                )
                .map_err(internal)?;
            page.objects_mut()
                .create_text_object(
                    PdfPoints::new(sidebar_x0 + 2.0 + TAG_PAD_X),
                    PdfPoints::new(fy(tag_y1 - 2.5)),
                    label.clone(),
                    font.clone(),
                    PdfPoints::new(TAG_FONT_SIZE),
                )
                .map_err(internal)?
                .set_fill_color(PdfColor::new(255, 255, 255, 255))
                .map_err(internal)?;
            next_tag_y = tag_y1 + TAG_GAP;
            count += 1;
        }
        Ok(count)
    }
}

fn pdfium_platform_dylib_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libpdfium.dylib"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "libpdfium.so"
    }
    #[cfg(target_os = "windows")]
    {
        "pdfium.dll"
    }
}
