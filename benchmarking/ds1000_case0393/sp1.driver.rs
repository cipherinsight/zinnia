// DS-1000 case 393 — SP1 host driver.
// Sample: a[i][j] = (i+1)*10 + (j+1); low=3, high=20 (clamped to 10). Result
// is columns j=3..=9 of every row.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const COL_SLICE_CLAMP_ELF: &[u8] = include_elf!("col-slice-clamp-program");

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

    const ROWS: usize = 20;
    const COLS: usize = 10;
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let mut row: Vec<f64> = Vec::with_capacity(COLS);
        for j in 0..COLS { row.push(((i + 1) * 10 + (j + 1)) as f64); }
        a.push(row);
    }
    let low: u32 = 3;
    let high_in: u32 = 20;
    let high: u32 = if high_in < (COLS as u32) { high_in } else { COLS as u32 };
    let width = (high - low) as usize;

    for row in a.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }
    stdin.write(&low);
    stdin.write(&high_in);
    for i in 0..ROWS {
        for k in 0..width {
            let v = a[i][(low as usize) + k];
            stdin.write(&v);
        }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(COL_SLICE_CLAMP_ELF);
        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).run().expect("failed to generate proof");
        println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
        proof.save("proof-with-pis-stark.bin").expect("saving proof failed");
        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).plonk().run().expect("failed to generate proof");
        println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
        proof.save("proof-with-pis.bin").expect("saving proof failed");
        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    }
}
