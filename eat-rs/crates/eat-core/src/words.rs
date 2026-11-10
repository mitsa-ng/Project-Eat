//! Shared OCR word model — identical schema to the Python pipeline's JSON:
//! `{"text": "word", "box": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]}`
//! All coordinates are in PDF point-space, origin = top-left.

use serde::{Deserialize, Serialize};

pub type PageIndex = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    #[serde(rename = "box")]
    pub box_: Box4,
}

/// Four-corner polygon, top-left origin (matches RapidOCR/PyMuPDF output).
pub type Box4 = [[f64; 2]; 4];

impl Word {
    /// Axis-aligned bounding rect (top-left origin): (x0, y0, x1, y1).
    pub fn rect(&self) -> (f64, f64, f64, f64) {
        let xs = [self.box_[0][0], self.box_[1][0], self.box_[2][0], self.box_[3][0]];
        let ys = [self.box_[0][1], self.box_[1][1], self.box_[2][1], self.box_[3][1]];
        (
            xs.iter().cloned().fold(f64::INFINITY, f64::min),
            ys.iter().cloned().fold(f64::INFINITY, f64::min),
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}

/// Page → words, preserving page order. Keys are page indices.
pub type Pages = Vec<Vec<Word>>;

/// Flatten pages into a single line of text using the script-appropriate
/// separator (no separator for zh/ja — port of `NLPEngine._words_to_text`).
pub fn words_to_text(words: &[Word], language: &str) -> String {
    let sep = if crate::languages::is_no_space(language) { "" } else { " " };
    words
        .iter()
        .filter(|w| !w.text.trim().is_empty())
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(sep)
}
