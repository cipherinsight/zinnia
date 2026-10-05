// DS-1000 case 335 — SP1 host driver. Three Float[50] arrays; element-wise max.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const MAX3_ELF: &[u8] = include_elf!("max-of-three-program");

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

    let mut a = [0.0f64; 50];
    let mut b = [0.0f64; 50];
    let mut c = [0.0f64; 50];
    for k in 0..50 {
        a[k] = 0.10 + 0.01 * k as f64;     // 0.10..0.59
        b[k] = 0.20 + 0.015 * k as f64;    // 0.20..0.935
        c[k] = 0.05 + 0.02 * k as f64;     // 0.05..1.03 — eventually max
    }
    let mut claimed = [0.0f64; 50];
    for k in 0..50 {
        let ab = if a[k] > b[k] { a[k] } else { b[k] };
        claimed[k] = if ab > c[k] { ab } else { c[k] };
    }

    for &v in a.iter() { stdin.write(&v); }
    for &v in b.iter() { stdin.write(&v); }
    for &v in c.iter() { stdin.write(&v); }
    for &v in claimed.iter() { stdin.write(&v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(MAX3_ELF);

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
