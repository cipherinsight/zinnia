// DS-1000 case 480 — SP1 host driver.
// x = [10, 20, 30, 40, 50, 60, 70, 80]; y = [1, 2, 3, 4, 5, 6, 7, 8].
// (a, b) = (50, 5) → match at index 4. Expected result = 4.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PRED_ARGMAX_ELF: &[u8] = include_elf!("pred-argmax-program");

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

    let x: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let y: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let a: i64 = 50;
    let b: i64 = 5;
    let claimed: i64 = 4;

    for &v in x.iter() { stdin.write(&v); }
    for &v in y.iter() { stdin.write(&v); }
    stdin.write(&a);
    stdin.write(&b);
    stdin.write(&claimed);

    if args.execute {
        let (_o, report) = client.execute(PRED_ARGMAX_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PRED_ARGMAX_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis-stark.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).plonk().run().unwrap();
    println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
}
