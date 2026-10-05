// DS-1000 case 0497 — SP1 host driver.
// a[i][j] = (i+1)*0.1 + (j+1)*0.01. shape = (45, 12). element = 7.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PAD2D_ELF: &[u8] = include_elf!("pad2d-program");

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
    const R: usize = 42;
    const C: usize = 10;
    let new_rows: u32 = 45;
    let new_cols: u32 = 12;
    let element: i64 = 7;
    let mut a = [[0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = (i as f64 + 1.0) * 0.1 + (j as f64 + 1.0) * 0.01; } }
    stdin.write(&new_rows);
    stdin.write(&new_cols);
    stdin.write(&element);
    for row in &a { for &v in row { stdin.write(&v); } }
    let pad = element as f64;
    for i in 0..new_rows as usize {
        for j in 0..new_cols as usize {
            let v: f64 = if i < R && j < C { a[i][j] } else { pad };
            stdin.write(&v);
        }
    }
    if args.execute {
        let (_o, report) = client.execute(PAD2D_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PAD2D_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
