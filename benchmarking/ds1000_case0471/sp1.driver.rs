// DS-1000 case 0471 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const LAST_SLICE_ELF: &[u8] = include_elf!("last-slice-program");

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
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 4;
    let mut z = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K {
        z[i][j][k] = (i as f64) * 100.0 + (j as f64) * 10.0 + (k as f64);
    } } }
    for row in &z { for col in row { for &v in col { stdin.write(&v); } } }
    for i in 0..N { for j in 0..M {
        let v: f64 = z[i][j][K - 1];
        stdin.write(&v);
    } }
    if args.execute {
        let (_o, report) = client.execute(LAST_SLICE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(LAST_SLICE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
