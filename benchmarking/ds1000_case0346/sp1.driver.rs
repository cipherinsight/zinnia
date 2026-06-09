// DS-1000 case 346 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ROWSCALE_ELF: &[u8] = include_elf!("rowscale-cumsum-program");

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

    // a[3-1] = a[2] = [1, 2, 3, 4, 5, 6]
    let data: [[f64; 6]; 8] = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ];
    let row: i64 = 3;
    let multiply_number: i64 = 2;
    // *2 -> [2,4,6,8,10,12], cumsum -> [2,6,12,20,30,42]
    let expected: [f64; 6] = [2.0, 6.0, 12.0, 20.0, 30.0, 42.0];

    for r in data.iter() {
        for &v in r.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&row);
    stdin.write(&multiply_number);
    for &v in expected.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(ROWSCALE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ROWSCALE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
