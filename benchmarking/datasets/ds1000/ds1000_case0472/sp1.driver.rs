// DS-1000 case 472 — SP1 host driver.
// Inputs filled deterministically with a[k] = (k as f64) * 0.01.
// Last slab (indices 7560..9072) is the expected output.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const SLAB_LAST_ELF: &[u8] = include_elf!("slab-last-program");

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

    const N: usize = 9072;
    const SLAB: usize = 1512;
    let mut a: Vec<f64> = Vec::with_capacity(N);
    for k in 0..N {
        a.push((k as f64) * 0.01);
    }
    for &v in a.iter() {
        stdin.write(&v);
    }
    let start = 5 * SLAB;
    for k in 0..SLAB {
        let v = a[start + k];
        stdin.write(&v);
    }

    if args.execute {
        let (_o, report) = client.execute(SLAB_LAST_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(SLAB_LAST_ELF);
    let start_t = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start_t.elapsed().as_millis());
    proof.save("proof-with-pis-stark.bin").unwrap();
    let start_t = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start_t.elapsed().as_millis());

    let start_t = Instant::now();
    let proof = client.prove(&pk, &stdin).plonk().run().unwrap();
    println!("Prove time (zk-SNARK) (ms): {:?}", start_t.elapsed().as_millis());
    proof.save("proof-with-pis.bin").unwrap();
    let start_t = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-SNARK) (ms): {:?}", start_t.elapsed().as_millis());
}
