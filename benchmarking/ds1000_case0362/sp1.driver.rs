// DS-1000 case 0362 — SP1 host driver.
// del_col = [1, 3, 5, 7]. mask drops 7. Keep cols (0-based) {1, 3, 5}. Output shape (8, 3).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DELCOL_ELF: &[u8] = include_elf!("delete-col-program");

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
    const R: usize = 8;
    const C: usize = 6;
    let mut a = [[0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = (i as f64 + 1.0) * 10.0 + (j as f64 + 1.0); } }
    let del_col: [i64; 4] = [1, 3, 5, 7];
    let out_cols: u32 = 3;
    stdin.write(&out_cols);
    for row in &a { for &v in row { stdin.write(&v); } }
    for &v in &del_col { stdin.write(&v); }
    let mut delete_mask = [false; C];
    for &v in &del_col {
        if v >= 1 && v <= C as i64 { delete_mask[(v - 1) as usize] = true; }
    }
    for i in 0..R {
        for j in 0..C {
            if !delete_mask[j] { stdin.write(&a[i][j]); }
        }
    }
    if args.execute {
        let (_o, report) = client.execute(DELCOL_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(DELCOL_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
