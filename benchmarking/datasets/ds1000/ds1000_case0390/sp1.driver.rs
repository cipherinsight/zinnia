// DS-1000 case 390 — SP1 host driver.
// Sample: a[i][j] = (i * 200 + j) as f64; patch_size = 10.
// Expected patches: patch p has block_col = p/10, block_row = p%10, and
// patch[r][c] = (block_row*10 + r) * 200 + (block_col*10 + c).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PATCH_RESHAPE_ELF: &[u8] = include_elf!("patch-reshape-program");

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
    const PS: usize = 10;
    let h_blocks = H / PS;
    let w_blocks = W / PS;
    let n_patches = h_blocks * w_blocks;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for i in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for j in 0..W { row.push((i * 200 + j) as f64); }
        a.push(row);
    }
    let patch_size: u32 = PS as u32;

    for row in a.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }
    stdin.write(&patch_size);

    for p in 0..n_patches {
        let block_col = p / h_blocks;
        let block_row = p % h_blocks;
        for r in 0..PS {
            for c in 0..PS {
                let v = ((block_row * PS + r) * 200 + (block_col * PS + c)) as f64;
                stdin.write(&v);
            }
        }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(PATCH_RESHAPE_ELF);
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
