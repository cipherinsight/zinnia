// DS-1000 case 302 — SP1 host driver.
// A = [1..20], nrow = 4; expected shape = (4, 5).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const RESHAPE_ELF: &[u8] = include_elf!("reshape-nrow-neg1-program");

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
    let nrow: i64 = 4;
    stdin.write(&nrow);
    let claim_nrow: i64 = 4;
    let claim_ncol: i64 = 5;
    stdin.write(&claim_nrow);
    stdin.write(&claim_ncol);
    for &v in a.iter() {
        stdin.write(&v);
    }
    let claim_total: i64 = 20;
    let claim_rank: i64 = 2;
    stdin.write(&claim_total);
    stdin.write(&claim_rank);

    if args.execute {
        let (_o, report) = client.execute(RESHAPE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(RESHAPE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
