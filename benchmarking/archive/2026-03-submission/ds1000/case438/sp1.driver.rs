// SP1 driver code
use alloy_sol_types::SolType;
use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci-program");

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long)] execute: bool,
    #[clap(long)] prove: bool,
}

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let args = Args::parse();
    if args.execute == args.prove {
        eprintln!("Error: You must specify either --execute or --prove");
        std::process::exit(1);
    }

    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    let post: [f32; 4] = [2.0, 5.0, 6.0, 10.0];
    let distance: [f32; 4] = [50.0, 100.0, 500.0, 1000.0];

    for i in 0..4 {
        stdin.write(&post[i]);
    }
    for i in 0..4 {
        stdin.write(&distance[i]);
    }

    // Compute expected Pearson correlation
    let n: f32 = 4.0;
    let mean_post = (post[0] + post[1] + post[2] + post[3]) / n;
    let mean_distance = (distance[0] + distance[1] + distance[2] + distance[3]) / n;

    let mut cov: f32 = 0.0;
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
    stdin.write(&pearson_r);

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(FIBONACCI_ELF);

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin)
            .run()
            .expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

        proof.save("proof-with-pis-stark.bin").expect("saving proof failed");

        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin)
            .plonk()
            .run()
            .expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-SNARK) (ms): {:?}", duration.as_millis());

        println!("Successfully generated proof!");

        proof.save("proof-with-pis.bin").expect("saving proof failed");

        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-SNARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");
    }
}
