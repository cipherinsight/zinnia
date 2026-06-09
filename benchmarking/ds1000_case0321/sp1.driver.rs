// DS-1000 case 321 — SP1 host driver.
// 5×6 matrix; maximum placed at a[2][4] → row=2, col=4.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ARGMAX_UNRAVEL_ELF: &[u8] = include_elf!("argmax-unravel-program");

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long)]
    execute: bool,
    #[clap(long)]
    prove: bool,
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

    let data: [[f64; 6]; 5] = [
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
        [0.31, 0.32, 0.33, 0.34, 99.0, 0.36],
        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46],
        [0.51, 0.52, 0.53, 0.54, 0.55, 0.56],
    ];
    for row in data.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    let claimed_row: u64 = 2;
    let claimed_col: u64 = 4;
    stdin.write(&claimed_row);
    stdin.write(&claimed_col);

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(ARGMAX_UNRAVEL_ELF);

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
