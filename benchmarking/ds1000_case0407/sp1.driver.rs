// DS-1000 case 0407 — SP1 host driver.
// a = [1..101], accmap = [(i%5)-2 for i in 0..100]. Output length = 3.
// Expected: [1010, 2000, 2040].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const BIN_NEG_ELF: &[u8] = include_elf!("bincount-negshift-program");

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
    let a: Vec<i64> = (1i64..=100).collect();
    let accmap: Vec<i64> = (0i64..100).map(|i| (i % 5) - 2).collect();
    let out_len: u32 = 3;
    for &v in &a { stdin.write(&v); }
    for &v in &accmap { stdin.write(&v); }
    stdin.write(&out_len);
    let expect: [i64; 3] = [1010, 2000, 2040];
    for &v in &expect { stdin.write(&v); }
    if args.execute {
        let (_o, report) = client.execute(BIN_NEG_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(BIN_NEG_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
