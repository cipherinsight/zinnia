// DS-1000 case 0291 -- Noir baseline. result = a.shape on Integer[2,5,5,2].
// Zinnia DCE eliminates loads of `a`; we mirror that here. Quantization
// is irrelevant (integer shape constants).

fn main(a: [Field; 100]) -> pub [Field; 4] {
    // `a` is unused by the kernel; we still take it as input so the
    // witness layout matches the other baselines.
    let _ = a[0];
    [2, 5, 5, 2]
}
