// DS-1000 case 453 — SP1 host driver.
// Rows chosen so ||row||₂ = 5 (Pythagorean): {3,4,0,0}, {0,3,4,0}, ...

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const L2_NORM_ELF: &[u8] = include_elf!("l2-norm-program");

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
        eprintln!("Error: specify either --execute or --prove");
        std::process::exit(1);
    }
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    let x: [[i64; 4]; 5] = [
        [3, 4, 0, 0],
        [0, 3, 4, 0],
        [0, 0, 3, 4],
        [4, 0, 0, 3],
        [3, 0, 4, 0],
    ];
    let mut claimed: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        let mut ss: f64 = 0.0;
        for j in 0..4 { ss += (x[i][j] as f64) * (x[i][j] as f64); }
        let l2 = ss.sqrt();
        for j in 0..4 { claimed[i][j] = (x[i][j] as f64) / l2; }
    }

    for row in x.iter() { for v in row.iter() { stdin.write(v); } }
    for row in claimed.iter() { for v in row.iter() { stdin.write(v); } }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(L2_NORM_ELF);
        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).run().expect("failed to generate proof");
        println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
        proof.save("proof-with-pis-stark.bin").expect("saving proof failed");
        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).plonk().run().expect("failed to generate proof");
        println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
        proof.save("proof-with-pis.bin").expect("saving proof failed");
        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    }
}
