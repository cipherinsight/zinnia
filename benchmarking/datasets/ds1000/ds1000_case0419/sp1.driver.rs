// DS-1000 case 419 — SP1 host driver.
// Reversed-then-binned-then-reversed: per row [base+1.5, base+3.5, base+5.5].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const REV_BIN_REV_MEAN_ELF: &[u8] = include_elf!("rev-bin-rev-mean-program");

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

    let data: [[f64; 6]; 8] = [
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        [41.0, 42.0, 43.0, 44.0, 45.0, 46.0],
        [51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
        [61.0, 62.0, 63.0, 64.0, 65.0, 66.0],
        [71.0, 72.0, 73.0, 74.0, 75.0, 76.0],
        [81.0, 82.0, 83.0, 84.0, 85.0, 86.0],
    ];
    let bin_size: u32 = 2;
    let expected: [[f64; 3]; 8] = [
        [11.5, 13.5, 15.5],
        [21.5, 23.5, 25.5],
        [31.5, 33.5, 35.5],
        [41.5, 43.5, 45.5],
        [51.5, 53.5, 55.5],
        [61.5, 63.5, 65.5],
        [71.5, 73.5, 75.5],
        [81.5, 83.5, 85.5],
    ];

    for row in data.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }
    stdin.write(&bin_size);
    for row in expected.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(REV_BIN_REV_MEAN_ELF);
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
