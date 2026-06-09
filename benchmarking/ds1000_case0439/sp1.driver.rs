// DS-1000 case 439 — SP1 host driver.
// X[i][k] = 0.01 * (i + 1) + 0.001 * (k + 1). claimed computed at host.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const OUTER_PROD_ELF: &[u8] = include_elf!("outer-prod-program");

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

    let mut x: [[f64; 20]; 10] = [[0.0; 20]; 10];
    for i in 0..10 {
        for k in 0..20 {
            x[i][k] = 0.01 * ((i + 1) as f64) + 0.001 * ((k + 1) as f64);
        }
    }
    // claimed[k][i][j] = x[i][k] * x[j][k].
    let mut claimed: Vec<f64> = Vec::with_capacity(20 * 10 * 10);
    for k in 0..20 {
        for i in 0..10 {
            for j in 0..10 {
                claimed.push(x[i][k] * x[j][k]);
            }
        }
    }

    for i in 0..10 {
        for k in 0..20 {
            stdin.write(&x[i][k]);
        }
    }
    for v in claimed.iter() { stdin.write(v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(OUTER_PROD_ELF);
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
