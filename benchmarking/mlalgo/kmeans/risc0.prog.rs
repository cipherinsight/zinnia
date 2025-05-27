use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let mut centroids = vec![vec![0.0; 2]; 10];
    for i in 0..3 {
        for j in 0..2 {
            centroids[i][j] = env::read();
        }
    }
    let mut classifications: Vec<u32> = Vec::new();
    for j in 0..10 {
        classifications.push(env::read());
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

    // write public output to the journal
    // env::commit(&input);
}
