// DS-1000 case 385 — SP1 host driver.
// Input a[i][j] = i*0.001 + j*0.0001 for determinism.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PATCH_ELF: &[u8] = include_elf!("patch-transpose-program");

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

    const H: usize = 100;
    const W: usize = 200;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for i in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for j in 0..W { row.push(i as f64 * 0.001 + j as f64 * 0.0001); }
        a.push(row);
    }

    // Expected: traverse the same way the kernel will.
    let mut expected: Vec<f64> = Vec::with_capacity(20000);
    for c_block in 0..100 {
        for r_block in 0..50 {
            for p in 0..2 {
                for q in 0..2 {
                    expected.push(a[r_block * 2 + p][c_block * 2 + q]);
                }
            }
        }
    }

    for row in a.iter() { for &v in row.iter() { stdin.write(&v); } }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(PATCH_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PATCH_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
