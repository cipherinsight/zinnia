// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    let n: usize = 27;
    let m: usize = 3;

    // read grades (27)
    let mut grades: Vec<f64> = Vec::new();
    for _ in 0..n {
        grades.push(env::read::<f64>());
    }

    // read eval (3)
    let mut evals: Vec<f64> = Vec::new();
    for _ in 0..m {
        evals.push(env::read::<f64>());
    }

    // read result (3)
    let mut result: Vec<f64> = Vec::new();
    for _ in 0..m {
        result.push(env::read::<f64>());
    }

    // 1) Verify non-decreasing sortedness
    for i in 0..(n - 1) {
        assert!(grades[i as usize] <= grades[(i + 1) as usize]);
    }

    // 2) Build ECDF table ys[i] = (i+1)/n
    let mut ys: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        ys[i as usize] = ((i + 1) as f64) / (n as f64);
    }

    // 3) Apply ECDF to evals as in the Zinnia logic
    let mut computed: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        let x: f64 = evals[i as usize];
        if x < grades[0] {
            computed[i as usize] = 0.0;
        } else if x >= grades[n - 1] {
            computed[i as usize] = 1.0;
        } else {
            // Find smallest j such that grades[j] > x
            let mut j: usize = 0;
            for k in 0..n {
                if grades[k as usize] > x {
                    j = k as usize;
                    break;
                }
            }
            computed[i as usize] = ys[j - 1];
        }
    }

    // np.allclose(computed, result) with default tolerances
    let rtol: f64 = 1e-08;
    let atol: f64 = 1e-08;
    for i in 0..m {
        let a = computed[i as usize];
        let b = result[i as usize];
        let ok = (a - b).abs() <= atol + rtol * b.abs();
        assert!(ok);
    }

    // env::commit(&output);
}
