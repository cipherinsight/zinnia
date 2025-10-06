fn max2(x: Field, y: Field) -> Field {
    // Return max(x, y)
    if x.lt(y) { y } else { x }
}

fn main(a: [Field; 3], b: [Field; 3], c: [Field; 3], result: [Field; 3]) {
    // Elementwise max across a, b, c
    for i in 0..3 {
        let m_ab = max2(a[i], b[i]);
        let m_abc = max2(m_ab, c[i]);
        assert(result[i] == m_abc, "Mismatch at index {i}");
    }
}
