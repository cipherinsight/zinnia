// DS-1000 case 347 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ROWDIV_ELF: &[u8] = include_elf!("rowdiv-product-program");

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

    // Row 2 -> [2, 4, 2, 2, 1, 1]; div=2 -> [1, 2, 1, 1, 0.5, 0.5]; product = 0.25.
    let data: [[f64; 6]; 8] = [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [2.0, 4.0, 2.0, 2.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ];
    let row: i64 = 3;
    let divide_number: i64 = 2;
    let expected: f64 = 0.25;

    for r in data.iter() {
        for &v in r.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&row);
    stdin.write(&divide_number);
    stdin.write(&expected);

    if args.execute {
        let (_o, report) = client.execute(ROWDIV_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ROWDIV_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
