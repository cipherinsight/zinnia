// DS-1000 case 361 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DELCOLS_ELF: &[u8] = include_elf!("delete-cols-6x6-program");

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

    let mut a: [[f64; 6]; 6] = [[0.0; 6]; 6];
    for i in 0..6 { for j in 0..6 { a[i][j] = (i as f64) * 10.0 + (j as f64) + 0.1; } }
    let kept: [usize; 4] = [1, 3, 4, 5];
    let mut expected: [[f64; 4]; 6] = [[0.0; 4]; 6];
    for i in 0..6 { for (jo, &j) in kept.iter().enumerate() { expected[i][jo] = a[i][j]; } }

    for r in a.iter() { for &v in r.iter() { stdin.write(&v); } }
    for r in expected.iter() { for &v in r.iter() { stdin.write(&v); } }

    if args.execute {
        let (_o, report) = client.execute(DELCOLS_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(DELCOLS_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
