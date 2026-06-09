// DS-1000 case 365 — SP1 host driver.
// Inputs: a[i] = (i+1)/10, pos=42, element=999. Expected: a with 999.0 inserted at index 42.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const INSERT_ELF: &[u8] = include_elf!("insert-program");

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

    let mut a: [f64; 100] = [0.0; 100];
    for i in 0..100 { a[i] = (i as f64 + 1.0) / 10.0; }
    let pos: u64 = 42;
    let element: i64 = 999;

    let mut expected: [f64; 101] = [0.0; 101];
    for i in 0..101 {
        expected[i] = if (i as u64) < pos {
            a[i]
        } else if (i as u64) == pos {
            element as f64
        } else {
            a[i - 1]
        };
    }

    for &v in a.iter() { stdin.write(&v); }
    stdin.write(&pos);
    stdin.write(&element);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(INSERT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(INSERT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
