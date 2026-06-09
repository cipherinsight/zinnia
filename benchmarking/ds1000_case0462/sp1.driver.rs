// DS-1000 case 462 — SP1 host driver.
// a = [0, 0, 1, 1, 2, 3, 3, 0, 4, 4, 5, 0, 6, 7, 7, 8, 0, 9, 9, 0]
// kept: drop dupes-against-prev AND drop zeros.
// i=0 a=0 dup_no, nz=0 → 0
// i=1 a=0 dup yes (==a[0]), nz=0 → 0
// i=2 a=1 not-dup, nz=1 → 1 ✓
// i=3 a=1 dup → 0
// i=4 a=2 ✓
// i=5 a=3 ✓
// i=6 a=3 dup → 0
// i=7 a=0 not-dup, nz=0 → 0
// i=8 a=4 ✓
// i=9 a=4 dup → 0
// i=10 a=5 ✓
// i=11 a=0 not-dup, nz=0 → 0
// i=12 a=6 ✓
// i=13 a=7 ✓
// i=14 a=7 dup → 0
// i=15 a=8 ✓
// i=16 a=0 not-dup, nz=0 → 0
// i=17 a=9 ✓
// i=18 a=9 dup → 0
// i=19 a=0 not-dup, nz=0 → 0
// kept = [1, 2, 3, 4, 5, 6, 7, 8, 9]

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DEDUPE_NZ_ELF: &[u8] = include_elf!("dedupe-nonzero-program");

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

    let a: [i64; 20] = [0, 0, 1, 1, 2, 3, 3, 0, 4, 4, 5, 0, 6, 7, 7, 8, 0, 9, 9, 0];
    let claimed: [i64; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let claimed_len: u64 = claimed.len() as u64;

    for v in a.iter() { stdin.write(v); }
    stdin.write(&claimed_len);
    for v in claimed.iter() { stdin.write(v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(DEDUPE_NZ_ELF);
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
