// DS-1000 case 300 — SP1 host driver.
// a = [1, 2, …, 20]; p = 25; expected percentile = 5.75.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PERCENTILE_ELF: &[u8] = include_elf!("percentile-program");

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

    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = (i as f64) + 1.0;
    }
    for &v in a.iter() {
        stdin.write(&v);
    }
    let p: i64 = 25;
    stdin.write(&p);
    let claim: f64 = 5.75;
    stdin.write(&claim);

    if args.execute {
        let (_o, report) = client.execute(PERCENTILE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PERCENTILE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
