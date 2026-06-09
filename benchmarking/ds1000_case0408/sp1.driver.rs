// DS-1000 case 0408 — SP1 host driver.
// a = [1..101], index = [(i%5)-2 for i in 0..100]. After shift: out_len=3. Expected: [98, 99, 100].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const GROUPMAX_NEG_ELF: &[u8] = include_elf!("groupmax-neg-program");

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args { #[clap(long)] execute: bool, #[clap(long)] prove: bool, }

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let args = Args::parse();
    if args.execute == args.prove { eprintln!("Error"); std::process::exit(1); }
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    let out_len: u32 = 3;
    let a: Vec<i64> = (1i64..=100).collect();
    let index: Vec<i64> = (0i64..100).map(|i| (i % 5) - 2).collect();
    stdin.write(&out_len);
    for &v in &a { stdin.write(&v); }
    for &v in &index { stdin.write(&v); }
    let expect: [i64; 3] = [98, 99, 100];
    for &v in &expect { stdin.write(&v); }
    if args.execute {
        let (_o, report) = client.execute(GROUPMAX_NEG_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(GROUPMAX_NEG_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
