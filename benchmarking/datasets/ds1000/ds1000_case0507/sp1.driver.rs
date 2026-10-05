// DS-1000 case 507 — SP1 host driver.
// Sample input is constructed so the bounding box is a fixed 3×3 region:
//   nonzeros confined to rows [1..3], cols [2..4]. The bbox shape and the
//   expected slice are baked into the driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const NONZERO_BBOX_ELF: &[u8] = include_elf!("nonzero-bbox-5x6-program");

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

    // 5×6 input: zero everywhere except a 3×3 block at rows 1..3, cols 2..4.
    let im: [[i64; 6]; 5] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 0],
        [0, 0, 4, 5, 6, 0],
        [0, 0, 7, 8, 9, 0],
        [0, 0, 0, 0, 0, 0],
    ];
    let rmin: u64 = 1;
    let rmax: u64 = 3;
    let cmin: u64 = 2;
    let cmax: u64 = 4;
    let expected: [i64; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

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
        let (_o, report) = client.execute(NONZERO_BBOX_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(NONZERO_BBOX_ELF);

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
