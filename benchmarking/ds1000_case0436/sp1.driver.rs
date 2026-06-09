// DS-1000 case 436 — SP1 host driver.
// Sample: a[i][j] = (i+1)*10 + (j+1)*0.1 → row max at j=4 for each row.
// To place the max in a non-trivial position, bump a[i][2] = a[i][4] + 1.0
// for each row, so per-row max is at j=2.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ROW_MAX_MASK_ELF: &[u8] = include_elf!("row-max-mask-program");

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

    let mut a: [f64; 50] = [0.0; 50];
    for i in 0..10 {
        for j in 0..5 {
            a[i * 5 + j] = (i as f64 + 1.0) * 10.0 + (j as f64 + 1.0) * 0.1;
        }
        // Make per-row max at column 2.
        a[i * 5 + 2] = (i as f64 + 1.0) * 10.0 + 99.0;
    }
    let mut expected: [u32; 50] = [0; 50];
    for i in 0..10 { expected[i * 5 + 2] = 1; }

    for &v in a.iter() { stdin.write(&v); }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(ROW_MAX_MASK_ELF);
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
