// DS-1000 case 467 — SP1 host driver.
// Sample input: a[i][j] = i*4 + j + 1 (4×4 ints), size = (3,3).
// Expected: 16 windows; per-window counts and full flattened buffer below.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const WINDOW_ROW_ELF: &[u8] = include_elf!("window-row-program");

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

    let a: [[i64; 4]; 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];
    let size: [i64; 2] = [3, 3];
    for row in a.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&size[0]);
    stdin.write(&size[1]);

    let counts: [u64; 16] = [4, 6, 6, 4, 6, 9, 9, 6, 6, 9, 9, 6, 4, 6, 6, 4];
    let total: u64 = 100;
    let flat: [i64; 100] = [
        1, 2, 5, 6,
        1, 2, 3, 5, 6, 7,
        2, 3, 4, 6, 7, 8,
        3, 4, 7, 8,
        1, 2, 5, 6, 9, 10,
        1, 2, 3, 5, 6, 7, 9, 10, 11,
        2, 3, 4, 6, 7, 8, 10, 11, 12,
        3, 4, 7, 8, 11, 12,
        5, 6, 9, 10, 13, 14,
        5, 6, 7, 9, 10, 11, 13, 14, 15,
        6, 7, 8, 10, 11, 12, 14, 15, 16,
        7, 8, 11, 12, 15, 16,
        9, 10, 13, 14,
        9, 10, 11, 13, 14, 15,
        10, 11, 12, 14, 15, 16,
        11, 12, 15, 16,
    ];
    for &c in counts.iter() {
        stdin.write(&c);
    }
    stdin.write(&total);
    for &v in flat.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(WINDOW_ROW_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(WINDOW_ROW_ELF);

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
