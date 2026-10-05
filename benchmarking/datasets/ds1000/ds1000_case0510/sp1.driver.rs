// DS-1000 case 510 — SP1 host driver.
// Sample input: 10×10 Float matrix that is zero everywhere except a 4×4 block
// of nonzeros at rows 3..6, cols 2..5. The bbox is rows [3..6], cols [2..5]
// (shape 4×4).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const NONZERO_BBOX_F64_ELF: &[u8] = include_elf!("nonzero-bbox-10x10-program");

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

    let mut im = [[0f64; 10]; 10];
    let mut counter: usize = 1;
    for di in 0..4 {
        for dj in 0..4 {
            im[3 + di][2 + dj] = (counter as f64) / 10.0;
            counter += 1;
        }
    }

    let rmin: u64 = 3;
    let rmax: u64 = 6;
    let cmin: u64 = 2;
    let cmax: u64 = 5;
    let mut expected: Vec<f64> = Vec::with_capacity(16);
    for i in (rmin as usize)..=(rmax as usize) {
        for j in (cmin as usize)..=(cmax as usize) {
            expected.push(im[i][j]);
        }
    }

    for row in im.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&rmin);
    stdin.write(&rmax);
    stdin.write(&cmin);
    stdin.write(&cmax);
    for &v in expected.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(NONZERO_BBOX_F64_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(NONZERO_BBOX_F64_ELF);

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
