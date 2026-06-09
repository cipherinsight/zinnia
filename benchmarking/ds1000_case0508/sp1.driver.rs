// DS-1000 case 508 — SP1 host driver.
// Sample input: zero matrix with a 4×4 nonzero block at rows 2..5, cols 3..6.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ARGWHERE_BBOX_ELF: &[u8] = include_elf!("argwhere-bbox-10x10-program");

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

    let mut a = [[0i64; 10]; 10];
    let block: [[i64; 4]; 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];
    for di in 0..4 {
        for dj in 0..4 {
            a[2 + di][3 + dj] = block[di][dj];
        }
    }
    let ystart: u64 = 2;
    let ystop: u64 = 6;
    let xstart: u64 = 3;
    let xstop: u64 = 7;
    let expected: [i64; 16] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ];

    for row in a.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&ystart);
    stdin.write(&ystop);
    stdin.write(&xstart);
    stdin.write(&xstop);
    for &v in expected.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(ARGWHERE_BBOX_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(ARGWHERE_BBOX_ELF);

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
