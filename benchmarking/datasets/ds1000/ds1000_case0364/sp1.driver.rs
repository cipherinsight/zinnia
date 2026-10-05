// DS-1000 case 364 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const INSERT2D_ELF: &[u8] = include_elf!("insert-row-2d-program");

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

    let mut a = [[0.0f64; 10]; 100];
    for i in 0..100 { for j in 0..10 { a[i][j] = (i as f64) + (j as f64) * 0.01; } }
    let pos: i64 = 5;
    let mut element = [0.0f64; 10];
    for j in 0..10 { element[j] = 999.0 + (j as f64); }

    let mut expected = [[0.0f64; 10]; 101];
    for i in 0..101 {
        if i < pos as usize {
            for j in 0..10 { expected[i][j] = a[i][j]; }
        } else if i == pos as usize {
            for j in 0..10 { expected[i][j] = element[j]; }
        } else {
            for j in 0..10 { expected[i][j] = a[i - 1][j]; }
        }
    }

    for r in a.iter() { for &v in r.iter() { stdin.write(&v); } }
    stdin.write(&pos);
    for &v in element.iter() { stdin.write(&v); }
    for r in expected.iter() { for &v in r.iter() { stdin.write(&v); } }

    if args.execute {
        let (_o, report) = client.execute(INSERT2D_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(INSERT2D_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
