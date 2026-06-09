// DS-1000 case 0299 — SP1 host driver.
// a[i][j] = ((i*20+j) % 4) + 5. raveled length 200. cols=4.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ONEHOT_2D_ELF: &[u8] = include_elf!("onehot-2d-program");

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
    let flat: Vec<i64> = (0i64..200).map(|k| (k % 4) + 5).collect();
    let cols: u32 = 4;
    stdin.write(&cols);
    for &v in &flat { stdin.write(&v); }
    let m = *flat.iter().min().unwrap();
    let temp: Vec<i64> = flat.iter().map(|v| v - m).collect();
    for i in 0..200usize {
        for c in 0..cols as usize {
            let v: u64 = if c == temp[i] as usize { 1 } else { 0 };
            stdin.write(&v);
        }
    }
    if args.execute {
        let (_o, report) = client.execute(ONEHOT_2D_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ONEHOT_2D_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
