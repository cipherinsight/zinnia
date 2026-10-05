// DS-1000 case 405 — SP1 host driver.
// a = [1..101], accmap = [0,1,2,3,4, 0,1,2,3,4, …]. Output: [970, 990, 1010, 1030, 1050].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const BINCOUNT_ELF: &[u8] = include_elf!("bincount-program");

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
    if args.execute == args.prove { eprintln!("Error"); std::process::exit(1); }
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    let n: u32 = 100;
    let out_len: u32 = 5;
    let a: Vec<i64> = (1i64..=100).collect();
    let accmap: Vec<i64> = (0i64..100).map(|i| i % 5).collect();
    stdin.write(&n);
    stdin.write(&out_len);
    for &v in &a { stdin.write(&v); }
    for &v in &accmap { stdin.write(&v); }
    let mut result = vec![0i64; out_len as usize];
    for i in 0..n as usize { result[accmap[i] as usize] += a[i]; }
    for v in &result { stdin.write(v); }
    if args.execute {
        let (_o, report) = client.execute(BINCOUNT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(BINCOUNT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
