// DS-1000 case 374 — SP1 guest program.
// Kernel: ecdf_result returns (xs, ys) for grades, then for each `eval[i]`:
//   if element < xs[0]: 0
//   elif element >= xs[-1]: 1
//   else: ys[(xs > element).argmax() - 1]
// Result is Float[5].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: usize = 50;
    let m: usize = 5;
    let mut grades: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        grades.push(sp1_zkvm::io::read::<f64>());
    }
    let mut eval: Vec<i64> = Vec::with_capacity(m);
    for _ in 0..m {
        eval.push(sp1_zkvm::io::read::<i64>());
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(m);
    for _ in 0..m {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    // Sort grades into xs.
    let mut xs: Vec<f64> = grades.clone();
    for i in 1..n {
        let key = xs[i];
        let mut j: usize = i;
        while j > 0 && xs[j - 1] > key {
            xs[j] = xs[j - 1];
            j -= 1;
        }
        xs[j] = key;
    }
    let nf: f64 = n as f64;
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        ys.push((i as f64 + 1.0) / nf);
    }

    let mut result: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        let element: f64 = eval[i] as f64;
        if element < xs[0] {
            result[i] = 0.0;
        } else if element >= xs[n - 1] {
            result[i] = 1.0;
        } else {
            // (xs > element).argmax() — index of first xs[k] > element.
            // First-occurrence semantics for argmax of a boolean array.
            let mut best_idx: usize = 0;
            let mut found: bool = false;
            for k in 0..n {
                if !found && xs[k] > element {
                    best_idx = k;
                    found = true;
                }
            }
            result[i] = ys[best_idx - 1];
        }
    }

    for i in 0..m {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
