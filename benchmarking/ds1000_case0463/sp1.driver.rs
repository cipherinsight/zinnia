// DS-1000 case 463 — SP1 host driver.
// Sample input matches the DS-1000 reference example padded to length 20.
// a = [0,0,1,1,1,2,2,0,1,3,3,3,0,0,5,5,4,4,0,7]
// Expected compact result: [1,2,1,3,5,4,7], length 7.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DEDUP_NONZERO_ELF: &[u8] = include_elf!("dedup-nonzero-program");

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

    let a: [i64; 20] = [
        0, 0, 1, 1, 1, 2, 2, 0, 1, 3, 3, 3, 0, 0, 5, 5, 4, 4, 0, 7,
    ];
    for &v in a.iter() {
        stdin.write(&v);
    }
    let claimed_len: u64 = 7;
    let mut claimed: [i64; 20] = [0; 20];
    let expected: [i64; 7] = [1, 2, 1, 3, 5, 4, 7];
    for k in 0..7 {
        claimed[k] = expected[k];
    }
    stdin.write(&claimed_len);
    for &v in claimed.iter() {
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(DEDUP_NONZERO_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(DEDUP_NONZERO_ELF);

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
