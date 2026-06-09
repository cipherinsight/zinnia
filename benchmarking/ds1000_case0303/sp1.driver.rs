// DS-1000 case 303 — SP1 host driver.
// Sample: A = [1..23] (as floats), ncol = 5 → col_keep = 20, rows = 4,
// output keeps a[0..20] and zeroes a[20..23].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const RESHAPE_TRIM_ELF: &[u8] = include_elf!("reshape-trim-23-program");

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

    let mut a: [f64; 23] = [0.0; 23];
    for i in 0..23 {
        a[i] = (i as f64) + 1.0;
    }
    for &v in a.iter() {
        stdin.write(&v);
    }
    let ncol: i64 = 5;
    stdin.write(&ncol);

    // Expected output: a[0..20] preserved, a[20..23] zeroed.
    let mut out: [f64; 23] = [0.0; 23];
    let col_keep: i64 = 20;
    for i in 0..23 {
        out[i] = if (i as i64) < col_keep { a[i] } else { 0.0 };
    }
    for &v in out.iter() {
        stdin.write(&v);
    }
    let claim_rows: i64 = 4;
    let claim_ncol: i64 = 5;
    let claim_total: i64 = 20;
    let claim_rank: i64 = 2;
    stdin.write(&claim_rows);
    stdin.write(&claim_ncol);
    stdin.write(&claim_total);
    stdin.write(&claim_rank);

    if args.execute {
        let (_o, report) = client.execute(RESHAPE_TRIM_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(RESHAPE_TRIM_ELF);
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
