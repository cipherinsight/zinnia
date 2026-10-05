// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read inputs
    let mut post: [f32; 4] = [0.0; 4];
    let mut distance: [f32; 4] = [0.0; 4];
    for i in 0..4 {
        post[i] = env::read();
    }
    for i in 0..4 {
        distance[i] = env::read();
    }

    let result: f32 = env::read();

    let n: f32 = 4.0;
    let mean_post = (post[0] + post[1] + post[2] + post[3]) / n;
    let mean_distance = (distance[0] + distance[1] + distance[2] + distance[3]) / n;

    let mut cov = 0.0;
    for i in 0..4 {
        cov += (post[i] - mean_post) * (distance[i] - mean_distance);
    }
    cov /= n;

    let mut var_post = 0.0;
    let mut var_distance = 0.0;
    for i in 0..4 {
        var_post += (post[i] - mean_post).powf(2.0);
        var_distance += (distance[i] - mean_distance).powf(2.0);
    }
    var_post /= n;
    var_distance /= n;

    let std_post = var_post.sqrt();
    let std_distance = var_distance.sqrt();

    let pearson_r = cov / (std_post * std_distance);

    assert!((pearson_r - result).abs() < 1e-6);
}
