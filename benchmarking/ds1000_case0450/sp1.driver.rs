// DS-1000 case 450 — SP1 host driver.
// a[i][j][k] = i*100 + j*10 + k (encoded as f64). second = [2, 5]; third = [3, 7].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const FANCY_IDX_ELF: &[u8] = include_elf!("fancy-index-program");

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

    let mut a: [[[f64; 9]; 8]; 7] = [[[0.0; 9]; 8]; 7];
    for i in 0..7 {
        for j in 0..8 {
            for k in 0..9 {
                a[i][j][k] = (i * 100 + j * 10 + k) as f64;
            }
        }
    }
    let second: [i64; 2] = [2, 5];
    let third: [i64; 2] = [3, 7];
    let mut claimed: [[[f64; 2]; 2]; 7] = [[[0.0; 2]; 2]; 7];
    for i in 0..7 {
        for p in 0..2 {
            for q in 0..2 {
                let j = second[p] as usize;
                let k = third[q] as usize;
                claimed[i][p][q] = a[i][j][k];
            }
        }
    }

    for i in 0..7 {
        for j in 0..8 {
            for k in 0..9 {
                stdin.write(&a[i][j][k]);
            }
        }
    }
    for v in second.iter() { stdin.write(v); }
    for v in third.iter() { stdin.write(v); }
    for plane in claimed.iter() {
        for row in plane.iter() {
            for &v in row.iter() { stdin.write(&v); }
        }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(FANCY_IDX_ELF);
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
