// DS-1000 case 429 — SP1 host driver.
// Sample a = [1..30]; mean=15.5, std≈8.6554, mean±2·std ≈ (-1.811, 32.811).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const MEAN_STD_2S_ELF: &[u8] = include_elf!("mean-std-2sigma-program");

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

    let mut a: [f64; 30] = [0.0; 30];
    for i in 0..30 { a[i] = (i + 1) as f64; }
    let claimed_lo: f64 = -1.810882896798379;
    let claimed_hi: f64 = 32.810882896798375;
    for &v in a.iter() { stdin.write(&v); }
    stdin.write(&claimed_lo);
    stdin.write(&claimed_hi);

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(MEAN_STD_2S_ELF);
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
