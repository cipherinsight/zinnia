// DS-1000 case 485 — SP1 host driver.
// A[i][j][p] = (i*18 + j*3 + p) * 0.1; B[p][k] = (p*3 + k + 1) * 0.5.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const TENSORDOT_ELF: &[u8] = include_elf!("tensordot-program");

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

    for i in 0..5 {
        for j in 0..6 {
            for p in 0..3 {
                let v: f64 = ((i * 18 + j * 3 + p) as f64) * 0.1;
                stdin.write(&v);
            }
        }
    }
    for p in 0..3 {
        for k in 0..3 {
            let v: f64 = ((p * 3 + k + 1) as f64) * 0.5;
            stdin.write(&v);
        }
    }
    let expected: [f64; 90] = [
        0.9, 1.05, 1.2, 2.7, 3.3, 3.9, 4.5, 5.55, 6.6, 6.3, 7.8, 9.3, 8.1, 10.05, 12.0, 9.9, 12.3, 14.7,
        11.7, 14.55, 17.4, 13.5, 16.8, 20.1, 15.3, 19.05, 22.8, 17.1, 21.3, 25.5, 18.9, 23.55, 28.2, 20.7, 25.8, 30.9,
        22.5, 28.05, 33.6, 24.3, 30.3, 36.3, 26.1, 32.55, 39.0, 27.9, 34.8, 41.7, 29.7, 37.05, 44.4, 31.5, 39.3, 47.1,
        33.3, 41.55, 49.8, 35.1, 43.8, 52.5, 36.9, 46.05, 55.2, 38.7, 48.3, 57.9, 40.5, 50.55, 60.6, 42.3, 52.8, 63.3,
        44.1, 55.05, 66.0, 45.9, 57.3, 68.7, 47.7, 59.55, 71.4, 49.5, 61.8, 74.1, 51.3, 64.05, 76.8, 53.1, 66.3, 79.5,
    ];
    for &v in expected.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(TENSORDOT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(TENSORDOT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis-stark.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).plonk().run().unwrap();
    println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
}
