// DS-1000 case 0366 — SP1 host driver.
// a[i][j] = (i+1)*0.1 + (j+1)*0.01. pos = [10,20,30,40,50] → adjusted [10,19,28,37,46].
// element[k][j] = 1000 + k*10 + j. Output 105 rows.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const INSERT_ELF: &[u8] = include_elf!("insert-rows-program");

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
    const R: usize = 100;
    const C: usize = 10;
    const E: usize = 5;
    let mut a = vec![vec![0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = (i as f64 + 1.0) * 0.1 + (j as f64 + 1.0) * 0.01; } }
    let pos: [i64; E] = [10, 20, 30, 40, 50];
    let mut element = [[0.0f64; C]; E];
    for k in 0..E { for j in 0..C { element[k][j] = 1000.0 + (k as f64) * 10.0 + (j as f64); } }
    for row in &a { for &v in row { stdin.write(&v); } }
    for &v in &pos { stdin.write(&v); }
    for row in &element { for &v in row { stdin.write(&v); } }
    let adjusted: [i64; E] = std::array::from_fn(|k| pos[k] - (k as i64));
    let mut out: Vec<[f64; C]> = Vec::with_capacity(R + E);
    let mut k = 0usize;
    for src in 0..R {
        while k < E && adjusted[k] as usize == src { out.push(element[k]); k += 1; }
        let mut row = [0.0f64; C];
        for j in 0..C { row[j] = a[src][j]; }
        out.push(row);
    }
    while k < E { out.push(element[k]); k += 1; }
    for row in &out { for &v in row { stdin.write(&v); } }
    if args.execute {
        let (_o, report) = client.execute(INSERT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(INSERT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
