//! Image operations for eat-ocr — ports of the cv2 operations used by
//! rapidocr's preprocessing (resize, dilate, perspective crop, rotations).

use crate::geom::{apply_h, homography, Pt};

/// 8-bit BGR image buffer (PP-OCR models were trained on cv2/BGR inputs).
#[derive(Clone)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>, // BGRBGR…
}

impl Img {
    pub fn from_bgr(w: usize, h: usize, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), w * h * 3);
        Img { w, h, data }
    }

    pub fn from_rgb(w: usize, h: usize, rgb: &[u8]) -> Self {
        let mut data = vec![0u8; w * h * 3];
        for i in 0..w * h {
            data[i * 3] = rgb[i * 3 + 2];
            data[i * 3 + 1] = rgb[i * 3 + 1];
            data[i * 3 + 2] = rgb[i * 3];
        }
        Img { w, h, data }
    }

    #[inline]
    pub fn px(&self, x: usize, y: usize) -> [u8; 3] {
        let o = (y * self.w + x) * 3;
        [self.data[o], self.data[o + 1], self.data[o + 2]]
    }

    /// Bilinear sample with BORDER_REPLICATE.
    fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let mut out = [0f32; 3];
        for c in 0..3 {
            let p00 = self.data[(y0 * self.w + x0) * 3 + c] as f32;
            let p10 = self.data[(y0 * self.w + x1) * 3 + c] as f32;
            let p01 = self.data[(y1 * self.w + x0) * 3 + c] as f32;
            let p11 = self.data[(y1 * self.w + x1) * 3 + c] as f32;
            out[c] = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;
        }
        out
    }

    /// Bilinear resize to (w, h) (≈ cv2.INTER_LINEAR for down/up-sampling).
    pub fn resize(&self, w: usize, h: usize) -> Img {
        if w == self.w && h == self.h {
            return self.clone();
        }
        let mut data = vec![0u8; w * h * 3];
        let sx = self.w as f32 / w as f32;
        let sy = self.h as f32 / h as f32;
        for y in 0..h {
            let src_y = (y as f32 + 0.5) * sy - 0.5;
            for x in 0..w {
                let src_x = (x as f32 + 0.5) * sx - 0.5;
                let p = self.sample_bilinear(src_x, src_y);
                let o = (y * w + x) * 3;
                for c in 0..3 {
                    data[o + c] = p[c].round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Img { w, h, data }
    }

    /// Rotate 180°.
    pub fn rotate180(&self) -> Img {
        let mut data = vec![0u8; self.data.len()];
        let n = self.w * self.h;
        for i in 0..n {
            let j = n - 1 - i;
            for c in 0..3 {
                data[i * 3 + c] = self.data[j * 3 + c];
            }
        }
        Img { w: self.w, h: self.h, data }
    }

    /// Rotate 90° counter-clockwise (np.rot90).
    pub fn rot90ccw(&self) -> Img {
        let (w, h) = (self.w, self.h);
        let (nw, nh) = (h, w);
        let mut data = vec![0u8; self.data.len()];
        for y in 0..h {
            for x in 0..w {
                // new[r][c] = old[c][W-1-r]  →  for old(x,y): r = W-1-x, c = y
                let r = w - 1 - x;
                let c = y;
                let dst = (r * nw + c) * 3;
                let src = (y * w + x) * 3;
                data[dst..dst + 3].copy_from_slice(&self.data[src..src + 3]);
            }
        }
        Img { w: nw, h: nh, data }
    }

    /// Perspective crop — port of rapidocr get_rotate_crop_image: map the
    /// quadrilateral `points` (tl, tr, br, bl) onto an axis-aligned crop,
    /// sampling with bilinear interpolation and BORDER_REPLICATE, then rotate
    /// 90° ccw when the crop is taller than wide × 1.5.
    pub fn perspective_crop(&self, points: [Pt; 4]) -> Img {
        let crop_w = points[1].sub(points[0]).norm().max(points[2].sub(points[3]).norm()) as usize;
        let crop_h = points[3].sub(points[0]).norm().max(points[2].sub(points[1]).norm()) as usize;
        if crop_w == 0 || crop_h == 0 {
            return Img { w: 1, h: 1, data: vec![0, 0, 0] };
        }
        let dst_std = [
            Pt::new(0.0, 0.0),
            Pt::new(crop_w as f64, 0.0),
            Pt::new(crop_w as f64, crop_h as f64),
            Pt::new(0.0, crop_h as f64),
        ];
        // inverse mapping: dst → src (what warpPerspective effectively samples)
        let h = homography(dst_std, points).unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut data = vec![0u8; crop_w * crop_h * 3];
        for y in 0..crop_h {
            for x in 0..crop_w {
                let (sx, sy) = apply_h(&h, x as f64 + 0.5, y as f64 + 0.5);
                let p = self.sample_bilinear(sx as f32 - 0.5, sy as f32 - 0.5);
                let o = (y * crop_w + x) * 3;
                for c in 0..3 {
                    data[o + c] = p[c].round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        let img = Img { w: crop_w, h: crop_h, data };
        if img.h as f64 / img.w as f64 >= 1.5 {
            img.rot90ccw()
        } else {
            img
        }
    }
}

/// 2×2 dilation matching cv2's even-kernel default anchor — a source pixel
/// lights itself and its right/down neighbours, so out(x,y) checks the
/// (x-1..=x, y-1..=y) neighbourhood (empirically verified against cv2).
pub fn dilate2x2(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let mut on = 0u8;
            'outer: for yy in y.saturating_sub(1)..=y {
                for xx in x.saturating_sub(1)..=x {
                    if mask[yy * w + xx] != 0 {
                        on = 1;
                        break 'outer;
                    }
                }
            }
            out[y * w + x] = on;
        }
    }
    out
}
