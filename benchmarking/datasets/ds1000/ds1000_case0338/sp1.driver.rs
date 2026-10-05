// DS-1000 case 338 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DIAG_VSTACK_ELF: &[u8] = include_elf!("diag-vstack-program");

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

    let mut a = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            a[i][j] = (i as f64 + 1.0) * 10.0 + (j as f64 + 1.0);
        }
    }
    let mut claimed = [[0.0f64; 8]; 2];
    for i in 0..8 {
        claimed[0][i] = a[i][i];
        claimed[1][i] = a[i][7 - i];
    }

    for row in a.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }
    for row in claimed.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(DIAG_VSTACK_ELF);

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
