// DS-1000 case 433 — SP1 host driver.
// Sample: a[i][j] = i*4 + j + 1; zero_rows=2 (zero row 2); zero_cols=1 (zero col 1).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const SCALAR_ZERO_IDX_ELF: &[u8] = include_elf!("scalar-zero-idx-program");

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

    let mut a: [i64; 16] = [0; 16];
    for i in 0..4 { for j in 0..4 { a[i * 4 + j] = (i * 4 + j + 1) as i64; } }
    let zero_rows: i64 = 2;
    let zero_cols: i64 = 1;
    let mut expected: [i64; 16] = [0; 16];
    for i in 0..4 {
        for j in 0..4 {
            let row_match = (i as i64) == zero_rows;
            let col_match = (j as i64) == zero_cols;
            expected[i * 4 + j] = if row_match || col_match { 0 } else { a[i * 4 + j] };
        }
    }

    for &v in a.iter() { stdin.write(&v); }
    stdin.write(&zero_rows);
    stdin.write(&zero_cols);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(SCALAR_ZERO_IDX_ELF);
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
