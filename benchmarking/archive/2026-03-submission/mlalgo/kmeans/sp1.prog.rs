//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);

use alloy_sol_types::SolType;
use fibonacci_lib::PublicValuesStruct;

pub fn main() {
    // Read an input to the program.
    //
    // Behind the scenes, this compiles down to a custom system call which handles reading inputs
    // from the prover.
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        data.push(tmp);
    }
    let mut centroids = vec![vec![0.0; 2]; 10];
    for i in 0..3 {
        for j in 0..2 {
            centroids[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut classifications: Vec<u32> = Vec::new();
    for j in 0..10 {
        classifications.push(sp1_zkvm::io::read::<u32>());
    }

    let n = data.len();
    let classes = centroids.len();
    let mut labels = vec![0; n];

    for _ in 0..10 {
        // Assign each data point to the closest centroid
        for i in 0..n {
            let mut dists = vec![0.0; classes];
            for j in 0..classes {
                let dx = data[i][0] - centroids[j][0];
                let dy = data[i][1] - centroids[j][1];
                dists[j] = dx * dx + dy * dy;
            }
            labels[i] = dists
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
        }

        // Compute new centroids
        let mut new_centroids = vec![vec![0.0, 0.0]; classes];
        let mut counts = vec![0.0; classes];

        for i in 0..n {
            new_centroids[labels[i]][0] += data[i][0];
            new_centroids[labels[i]][1] += data[i][1];
            counts[labels[i]] += 1.0;
        }

        for i in 0..classes {
            if counts[i] > 0.0 {
                new_centroids[i][0] /= counts[i];
                new_centroids[i][1] /= counts[i];
            }
        }

        centroids = new_centroids;
    }

    for i in 0..10 {
        assert!(
            labels[i] == classifications[i] as usize,
            "Classification mismatch: expected {:?}, but got {:?}",
            classifications,
            labels
        );
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
