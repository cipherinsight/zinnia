// DS-1000 case 345 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const COLSCALE_ELF: &[u8] = include_elf!("colscale-cumsum-program");

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

    let data: [[f64; 6]; 8] = [
        [0.1, 0.2, 1.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 2.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 3.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 4.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 5.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 6.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 7.0, 0.4, 0.5, 0.6],
        [0.1, 0.2, 8.0, 0.4, 0.5, 0.6],
    ];
    let col: i64 = 3;            // col-1 == 2
    let multiply_number: i64 = 2; // doubles the column
    // After multiply column 2 -> [2,4,6,8,10,12,14,16], cumsum -> [2,6,12,20,30,42,56,72]
    let expected: [f64; 8] = [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];

    for row in data.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&col);
    stdin.write(&multiply_number);
    for &v in expected.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(COLSCALE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(COLSCALE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
