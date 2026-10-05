// DS-1000 case 509 — SP1 host driver.
// Sample input: 10×10 Float matrix with `im[i][j] = (i*10 + j + 1) / 10`,
// except four scattered zeros at (1,2), (1,6), (4,2), (4,6). This makes
//   rows = {1, 4}, cols = {2, 6}  →  bbox = im[1..5, 2..7], shape 4×5.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ZERO_BBOX_ELF: &[u8] = include_elf!("zero-bbox-10x10-program");

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

    let zeros: [(usize, usize); 4] = [(1, 2), (1, 6), (4, 2), (4, 6)];
    let mut im = [[0f64; 10]; 10];
    for i in 0..10 {
        for j in 0..10 {
            im[i][j] = ((i * 10 + j + 1) as f64) / 10.0;
        }
    }
    for &(i, j) in zeros.iter() {
        im[i][j] = 0.0;
    }

    let rmin: u64 = 1;
    let rmax: u64 = 4;
    let cmin: u64 = 2;
    let cmax: u64 = 6;
    // Expected slice: im[1..=4, 2..=6] → 4 rows × 5 cols = 20 entries.
    let mut expected: Vec<f64> = Vec::with_capacity(20);
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
        let (_o, report) = client.execute(ZERO_BBOX_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(ZERO_BBOX_ELF);

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
