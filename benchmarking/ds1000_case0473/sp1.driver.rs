// DS-1000 case 473 — SP1 host driver.
// c = [[[1,2]],[[3,4]],[[5,6]],[[7,8]]] (4×1×2), CNTS = [1, 5, 7].
// Result is always False (array_equal on (8,) vs scalar).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ANY_AEQ_ELF: &[u8] = include_elf!("any-aeq-program");

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

    let c: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let cnts: [i64; 3] = [1, 5, 7];
    let claimed: u64 = 0; // False

    for &v in c.iter() {
        stdin.write(&v);
    }
    for &v in cnts.iter() {
        stdin.write(&v);
    }
    stdin.write(&claimed);

    if args.execute {
        let (_o, report) = client.execute(ANY_AEQ_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ANY_AEQ_ELF);
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
