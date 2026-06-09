// DS-1000 case 0411 — SP1 host driver.
// a[i][j] = i*5 + j + 1. low=2, high=5 → 3×3 crop = [[13,14,15],[18,19,20],[23,24,25]].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const FILL_CROP_ELF: &[u8] = include_elf!("fill-crop-program");

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
    let low: i64 = 2;
    let high: i64 = 5;
    let mut a = [[0.0f64; 5]; 5];
    for i in 0..5 { for j in 0..5 { a[i][j] = (i as f64) * 5.0 + (j as f64) + 1.0; } }
    stdin.write(&low);
    stdin.write(&high);
    for row in &a { for &v in row { stdin.write(&v); } }
    let h = (high - low) as usize;
    for i in 0..h {
        for j in 0..h {
            let si = low + i as i64;
            let sj = low + j as i64;
            let v: f64 = if si >= 0 && si < 5 && sj >= 0 && sj < 5 {
                a[si as usize][sj as usize]
            } else { 0.0 };
            stdin.write(&v);
        }
    }
    if args.execute {
        let (_o, report) = client.execute(FILL_CROP_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(FILL_CROP_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
