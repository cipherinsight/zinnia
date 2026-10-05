// DS-1000 case 422 — SP1 host driver.
// Sample: a = [1..50], b = δ_{10} (zeros except b[10] = 1).
// c[i] = 1 only at i=9 (b[1:][9]=b[10]) and i=10+49=59 (b[10]).
// Closed-form: out[k] = a[k-40] (when 40≤k≤49) + a[k+10] (when 0≤k≤39).
// → out = [11, 12, ..., 50, 1, 2, ..., 10].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const CORRELATE_HSTACK_ELF: &[u8] = include_elf!("correlate-hstack-program");

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
    for i in 0..50 { a[i] = (i + 1) as f64; }
    let mut b: [f64; 50] = [0.0; 50];
    b[10] = 1.0;
    let mut expected: [f64; 50] = [0.0; 50];
    for k in 0..40 { expected[k] = (k + 11) as f64; }
    for k in 40..50 { expected[k] = (k - 39) as f64; }

    for &v in a.iter() { stdin.write(&v); }
    for &v in b.iter() { stdin.write(&v); }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(CORRELATE_HSTACK_ELF);
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
