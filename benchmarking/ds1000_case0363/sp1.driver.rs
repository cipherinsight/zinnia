// DS-1000 case 363 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const INSERT1D_ELF: &[u8] = include_elf!("insert-1d-program");

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

    let mut a = [0.0f64; 100];
    for i in 0..100 { a[i] = (i as f64) * 0.1; }
    let pos: i64 = 7;
    let element: i64 = 999;
    let mut expected = [0.0f64; 101];
    for i in 0..101 {
        if i < pos as usize { expected[i] = a[i]; }
        else if i == pos as usize { expected[i] = element as f64; }
        else { expected[i] = a[i - 1]; }
    }

    for &v in a.iter() { stdin.write(&v); }
    stdin.write(&pos);
    stdin.write(&element);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(INSERT1D_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(INSERT1D_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
