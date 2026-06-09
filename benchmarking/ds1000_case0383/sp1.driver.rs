// DS-1000 case 383 — SP1 host driver.
// N = 10 — top 10 indices descending.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ARGSORT_ELF: &[u8] = include_elf!("argsort-topn-program");

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

    let n: usize = 100;
    let big_n: u64 = 10;
    let m: usize = big_n as usize;
    let mut a: [f64; 100] = [0.0; 100];
    for i in 0..n {
        a[i] = ((i * 37 + 13) % 1000) as f64 + 0.5 * (i as f64);
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    let mut rev: Vec<u64> = (0..n).map(|i| idx[n - 1 - i] as u64).collect();
    let expected: Vec<u64> = rev.drain(..m).collect();

    for &v in a.iter() { stdin.write(&v); }
    stdin.write(&big_n);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(ARGSORT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ARGSORT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
