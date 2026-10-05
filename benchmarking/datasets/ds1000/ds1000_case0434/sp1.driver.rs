// DS-1000 case 434 — SP1 host driver.
// Sample: a[i][j] = i*4 + j + 1; zero_rows = [0, 2]; zero_cols = [1, 3].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ARRAY_ZERO_IDX_ELF: &[u8] = include_elf!("array-zero-idx-program");

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
    let zero_rows: [i64; 2] = [0, 2];
    let zero_cols: [i64; 2] = [1, 3];
    let mut expected: [i64; 16] = [0; 16];
    for i in 0..4 {
        let i_i = i as i64;
        let rz = (i_i == zero_rows[0]) || (i_i == zero_rows[1]);
        for j in 0..4 {
            let j_i = j as i64;
            let cz = (j_i == zero_cols[0]) || (j_i == zero_cols[1]);
            expected[i * 4 + j] = if rz || cz { 0 } else { a[i * 4 + j] };
        }
    }

    for &v in a.iter() { stdin.write(&v); }
    for &v in zero_rows.iter() { stdin.write(&v); }
    for &v in zero_cols.iter() { stdin.write(&v); }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(ARRAY_ZERO_IDX_ELF);
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
