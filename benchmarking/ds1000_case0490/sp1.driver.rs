// DS-1000 case 490 — SP1 host driver.
// arr[i][j][k] = ((i*16 + j*4 + k) % 30) as f64; n1 = [5,10,15,0,8]; n2 = [20,25,22,30,18].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const THRESH_ELF: &[u8] = include_elf!("threshold-program");

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

    let mut arr = [[[0.0f64; 4]; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            for k in 0..4 {
                arr[i][j][k] = ((i * 16 + j * 4 + k) % 30) as f64;
            }
        }
    }
    let n1: [i64; 5] = [5, 10, 15, 0, 8];
    let n2: [i64; 5] = [20, 25, 22, 30, 18];

    for i in 0..5 {
        for j in 0..4 {
            for k in 0..4 {
                stdin.write(&arr[i][j][k]);
            }
        }
    }
    for &v in n1.iter() { stdin.write(&v); }
    for &v in n2.iter() { stdin.write(&v); }

    let expected: [f64; 80] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 21.0, 22.0,
        23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 5.0, 6.0, 7.0, 8.0,
        0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 30.0, 30.0,
    ];
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(THRESH_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(THRESH_ELF);
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
