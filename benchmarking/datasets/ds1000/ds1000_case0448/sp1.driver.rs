// DS-1000 case 448 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const VSTACK_T_ELF: &[u8] = include_elf!("vstack-t-program");

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

    let x_dists: [[f64; 4]; 3] = [
        [0.11, 0.12, 0.13, 0.14],
        [0.21, 0.22, 0.23, 0.24],
        [0.31, 0.32, 0.33, 0.34],
    ];
    let y_dists: [[f64; 4]; 3] = [
        [1.11, 1.12, 1.13, 1.14],
        [1.21, 1.22, 1.23, 1.24],
        [1.31, 1.32, 1.33, 1.34],
    ];
    let mut claimed: [[[f64; 2]; 4]; 3] = [[[0.0; 2]; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            claimed[i][j][0] = x_dists[i][j];
            claimed[i][j][1] = y_dists[i][j];
        }
    }

    for row in x_dists.iter() { for &v in row.iter() { stdin.write(&v); } }
    for row in y_dists.iter() { for &v in row.iter() { stdin.write(&v); } }
    for plane in claimed.iter() {
        for row in plane.iter() {
            for &v in row.iter() { stdin.write(&v); }
        }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(VSTACK_T_ELF);
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
