//! Pythran make_circle_two_points: the incremental step of Welzl-style
//! smallest-enclosing-circle construction with one boundary point `p`.
//!
//! ```python
//! def _cross_product(x0, y0, x1, y1, x2, y2):
//!     return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
//!
//! def make_circumcircle(a, b, c):
//!     ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2.0
//!     oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2.0
//!     ax = a[0] - ox
//!     ay = a[1] - oy
//!     bx = b[0] - ox
//!     by = b[1] - oy
//!     cx = c[0] - ox
//!     cy = c[1] - oy
//!     d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
//!     if d == 0.0:
//!         return None
//!     x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
//!     y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
//!     ra = math.hypot(x - a[0], y - a[1])
//!     rb = math.hypot(x - b[0], y - b[1])
//!     rc = math.hypot(x - c[0], y - c[1])
//!     return (x, y, max(ra, rb, rc))
//!
//! def make_diameter(a, b):
//!     cx = (a[0] + b[0]) / 2.0
//!     cy = (a[1] + b[1]) / 2.0
//!     r0 = math.hypot(cx - a[0], cy - a[1])
//!     r1 = math.hypot(cx - b[0], cy - b[1])
//!     return (cx, cy, max(r0, r1))
//!
//! def is_in_circle(c, p):
//!     return math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * (1 + 1e-14)
//!
//! def make_circle_two_points(points, p, q):
//!     circ = make_diameter(p, q)
//!     left = None
//!     right = None
//!     px, py = p
//!     qx, qy = q
//!     for r in points:
//!         if is_in_circle(circ, r):
//!             continue
//!         cross = _cross_product(px, py, qx, qy, r[0], r[1])
//!         c = make_circumcircle(p, q, r)
//!         if c is None:
//!             continue
//!         elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
//!             left = c
//!         elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
//!             right = c
//!     if left is None and right is None:
//!         return circ
//!     elif left is None:
//!         return right
//!     elif right is None:
//!         return left
//!     else:
//!         return left if left[2] <= right[2] else right
//!
//! def _make_circle_one_point(points: Float[Array, '8 2'], p: Tuple[Float, Float]):
//!     c = (p[0], p[1], 0.0)
//!     for i, q in enumerate(points):
//!         if not is_in_circle(c, q):
//!             if c[2] == 0.0:
//!                 c = make_diameter(p, q)
//!             else:
//!                 c = make_circle_two_points(points[:i + 1], p, q)
//!     return c
//! ```
use crate::prelude::*;

/// A point `(x, y)`.
type Point = (Q32, Q32);
/// A circle `(x, y, radius)`.
type Circle = (Q32, Q32, Q32);

/// `math.hypot`: the root of the sum of squares (the squares summed raw and
/// rescaled once).
fn hypot(x: Q32, y: Q32) -> Q32 {
    dot([(x, x), (y, y)]).sqrt()
}

/// Twice the signed area of the triangle `(p, q, (x, y))`: positive when the
/// point is left of the line from `p` to `q`.
fn cross_product(p: Point, q: Point, x: Q32, y: Q32) -> Q32 {
    (q.0 - p.0) * (y - p.1) - (q.1 - p.1) * (x - p.0)
}

/// `is_in_circle`. The slack factor `1 + 1e-14` quantizes to exactly 1 in
/// Q32, so the test is `distance <= radius`.
fn is_in_circle(c: Circle, p: Point) -> bool {
    hypot(p.0 - c.0, p.1 - c.1) <= c.2
}

/// The circle with diameter `ab`.
fn make_diameter(a: Point, b: Point) -> Circle {
    let (cx, cy) = ((a.0 + b.0) / 2, (a.1 + b.1) / 2);
    let r0 = hypot(cx - a.0, cy - a.1);
    let r1 = hypot(cx - b.0, cy - b.1);
    (cx, cy, r0.max(r1))
}

/// The circle through `a`, `b`, `c` (coordinates taken relative to the
/// centre of their bounding box), or `None` if they are collinear.
fn make_circumcircle(a: Point, b: Point, c: Point) -> Option<Circle> {
    let ox = (a.0.min(b.0).min(c.0) + a.0.max(b.0).max(c.0)) / 2;
    let oy = (a.1.min(b.1).min(c.1) + a.1.max(b.1).max(c.1)) / 2;
    let (ax, ay) = (a.0 - ox, a.1 - oy);
    let (bx, by) = (b.0 - ox, b.1 - oy);
    let (cx, cy) = (c.0 - ox, c.1 - oy);
    let d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2;
    if d == Q32::ZERO {
        return None;
    }
    let (sa, sb, sc) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
    let x = ox + (sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)) / d;
    let y = oy + (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / d;
    let ra = hypot(x - a.0, y - a.1);
    let rb = hypot(x - b.0, y - b.1);
    let rc = hypot(x - c.0, y - c.1);
    Some((x, y, ra.max(rb).max(rc)))
}

/// The smallest circle through `p` and `q` enclosing `points`.
fn make_circle_two_points(points: ArrayView2<Q32>, p: Point, q: Point) -> Circle {
    let circ = make_diameter(p, q);
    let cross = |x: Q32, y: Q32| cross_product(p, q, x, y);
    let (mut left, mut right): (Option<Circle>, Option<Circle>) = (None, None);
    for row in points.rows() {
        let r = (row[0], row[1]);
        if is_in_circle(circ, r) {
            continue;
        }
        let side = cross(r.0, r.1);
        let Some(c) = make_circumcircle(p, q, r) else {
            continue;
        };
        if side > Q32::ZERO && left.is_none_or(|l| cross(c.0, c.1) > cross(l.0, l.1)) {
            left = Some(c);
        } else if side < Q32::ZERO && right.is_none_or(|rt| cross(c.0, c.1) < cross(rt.0, rt.1)) {
            right = Some(c);
        }
    }
    match (left, right) {
        (None, None) => circ,
        (None, Some(r)) => r,
        (Some(l), None) => l,
        (Some(l), Some(r)) => {
            if l.2 <= r.2 {
                l
            } else {
                r
            }
        }
    }
}

pub mod _make_circle_one_point {
    use super::*;

    /// The circle `(x, y, radius)`.
    pub type Output = (Q32, Q32, Q32);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let points: Array2<Q32> = args.array2();
        // `p: Tuple[Float, Float]` arrives as its two floats, in order.
        let p: Point = (args.scalar(), args.scalar());
        let mut c: Circle = (p.0, p.1, Q32::ZERO);
        for (i, row) in points.rows().into_iter().enumerate() {
            let q = (row[0], row[1]);
            if !is_in_circle(c, q) {
                c = if c.2 == Q32::ZERO {
                    make_diameter(p, q)
                } else {
                    make_circle_two_points(points.slice(s![..=i, ..]), p, q)
                };
            }
        }
        c
    }
}
