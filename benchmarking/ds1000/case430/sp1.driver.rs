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

    let a: [f32; 13] = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0, 10.0, 29.0, 32.0, 45.0];
    for v in a {
        stdin.write(&v);
    }

    let n: f32 = 13.0;
    let mut sum = 0.0;
    for v in a {
        sum += v;
    }
    let mean_val = sum / n;
    let mut var_sum = 0.0;
    for v in a {
        var_sum += (v - mean_val) * (v - mean_val);
    }
    let variance = var_sum / n;
    let std_val = variance.sqrt();
    let lower = mean_val - 3.0 * std_val;
    let upper = mean_val + 3.0 * std_val;

    stdin.write(&lower);
    stdin.write(&upper);

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
