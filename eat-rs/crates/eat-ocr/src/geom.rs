//! Geometry helpers for the DB text-detection post-process — ports of the
//! cv2/pyclipper/shapely operations in rapidocr's DBPostProcess.

/// 2-D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pt {
    pub x: f64,
    pub y: f64,
}

impl Pt {
    pub fn new(x: f64, y: f64) -> Self {
        Pt { x, y }
    }
    pub fn sub(self, o: Pt) -> Pt {
        Pt::new(self.x - o.x, self.y - o.y)
    }
    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// Andrew's monotone chain convex hull (input order independent).
pub fn convex_hull(points: &[Pt]) -> Vec<Pt> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    pts.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    let cross = |o: Pt, a: Pt, b: Pt| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut lower: Vec<Pt> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Pt> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area enclosing rectangle via rotating calipers over the convex
/// hull. Returns the 4 corners and (width, height) of the rect.
pub fn min_area_rect(points: &[Pt]) -> (Vec<Pt>, f64, f64) {
    let hull = convex_hull(points);
    if hull.len() < 2 {
        let p = hull.first().copied().unwrap_or(Pt::new(0.0, 0.0));
        return (vec![p; 4], 0.0, 0.0);
    }
    let n = hull.len();
    let mut best = (f64::INFINITY, Vec::new(), 0.0, 0.0); // (area, corners, w, h)
    for i in 0..n {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % n];
        let e = p2.sub(p1);
        let len = e.norm();
        if len < 1e-9 {
            continue;
        }
        let ux = Pt::new(e.x / len, e.y / len); // unit along edge
        let uy = Pt::new(-ux.y, ux.x); // unit perpendicular
        let mut min_a = f64::INFINITY;
        let mut max_a = f64::NEG_INFINITY;
        let mut min_b = f64::INFINITY;
        let mut max_b = f64::NEG_INFINITY;
        for &p in &hull {
            let d = p.sub(p1);
            let a = d.x * ux.x + d.y * ux.y;
            let b = d.x * uy.x + d.y * uy.y;
            min_a = min_a.min(a);
            max_a = max_a.max(a);
            min_b = min_b.min(b);
            max_b = max_b.max(b);
        }
        let w = max_a - min_a;
        let h = max_b - min_b;
        let area = w * h;
        if area < best.0 {
            let c = |a: f64, b: f64| Pt::new(p1.x + ux.x * a + uy.x * b, p1.y + ux.y * a + uy.y * b);
            let corners = vec![c(min_a, min_b), c(max_a, min_b), c(max_a, max_b), c(min_a, max_b)];
            best = (area, corners, w, h);
        }
    }
    (best.1, best.2, best.3)
}

/// Port of DBPostProcess.order_points_clockwise: tl, tr, br, bl.
pub fn order_points_clockwise(pts: [Pt; 4]) -> [Pt; 4] {
    let mut sorted = pts;
    sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    let mut left = [sorted[0], sorted[1]];
    let mut right = [sorted[2], sorted[3]];
    left.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
    right.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
    let (tl, bl) = (left[0], left[1]);
    let (tr, br) = (right[0], right[1]);
    [tl, tr, br, bl]
}

/// Port of DBPostProcess.get_mini_boxes: min-area rect corners reordered to a
/// stable rotation-aware starting point. Returns (4 points, min side length).
pub fn get_mini_boxes(points: &[Pt]) -> (Vec<Pt>, f64) {
    let (corners, w, h) = min_area_rect(points);
    let mut pts: Vec<Pt> = corners;
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let (i1, i4) = if pts[1].y > pts[0].y { (0, 1) } else { (1, 0) };
    let (i2, i3) = if pts[3].y > pts[2].y { (2, 3) } else { (3, 2) };
    let ordered = vec![pts[i1], pts[i2], pts[i3], pts[i4]];
    (ordered, w.min(h))
}

/// Expand a 4-corner polygon outward by `dist` along its rect axes — port of
/// DBPostProcess.unclip (pyclipper offset approximated by axis growth; corner
/// rounding differs only at the corners).
pub fn unclip_rect(points: &[Pt], unclip_ratio: f64) -> Vec<Pt> {
    let area = polygon_area(points);
    let length = polygon_perimeter(points);
    if length < 1e-6 {
        return points.to_vec();
    }
    let dist = area * unclip_ratio / length;
    // rect axes: u = p1-p0 (top edge), v = p3-p0 (left edge)
    let (u, ul) = {
        let e = points[1].sub(points[0]);
        let l = e.norm();
        (Pt::new(e.x / l, e.y / l), l)
    };
    let (v, vl) = {
        let e = points[3].sub(points[0]);
        let l = e.norm();
        (Pt::new(e.x / l, e.y / l), l)
    };
    let au = ul / 2.0 + dist;
    let av = vl / 2.0 + dist;
    let cx = (points[0].x + points[2].x) / 2.0;
    let cy = (points[0].y + points[2].y) / 2.0;
    let corner = |su: f64, sv: f64| Pt::new(cx + u.x * au * su + v.x * av * sv, cy + u.y * au * su + v.y * av * sv);
    vec![corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)]
}

pub fn polygon_area(p: &[Pt]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += p[i].x * p[j].y - p[j].x * p[i].y;
    }
    (a / 2.0).abs()
}

pub fn polygon_perimeter(p: &[Pt]) -> f64 {
    let n = p.len();
    let mut l = 0.0;
    for i in 0..n {
        l += p[(i + 1) % n].sub(p[i]).norm();
    }
    l
}

/// Even-odd point-in-polygon test (inclusive-ish; matches fillPoly closely
/// enough for a mean-score mask).
pub fn point_in_polygon(x: f64, y: f64, poly: &[Pt]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i].x, poly[i].y);
        let (xj, yj) = (poly[j].x, poly[j].y);
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Solve an 8×8 linear system by Gaussian elimination with partial pivoting
/// (used for the 4-point perspective transform).
pub fn solve_linear(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    let n = 8;
    for col in 0..n {
        // pivot
        let mut pivot = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let div = a[col][col];
        for c in col..n {
            a[col][c] /= div;
        }
        b[col] /= div;
        for r in 0..n {
            if r != col && a[r][col].abs() > 1e-15 {
                let f = a[r][col];
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some(*b)
}

/// Homography mapping `from` → `to` (4 point pairs) as a 3×3 row-major matrix.
pub fn homography(from: [Pt; 4], to: [Pt; 4]) -> Option<[f64; 9]> {
    let mut a = [[0.0; 8]; 8];
    let mut b = [0.0; 8];
    for i in 0..4 {
        let (x, y) = (from[i].x, from[i].y);
        let (u, v) = (to[i].x, to[i].y);
        a[i * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
        b[i * 2] = u;
        a[i * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
        b[i * 2 + 1] = v;
    }
    let h = solve_linear(&mut a, &mut b)?;
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

/// Apply a 3×3 homography to a point.
pub fn apply_h(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let d = h[6] * x + h[7] * y + h[8];
    ((h[0] * x + h[1] * y + h[2]) / d, (h[3] * x + h[4] * y + h[5]) / d)
}
