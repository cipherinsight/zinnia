// DS-1000 case 334 — SP1 host driver.
// a[i] = 0.30 + 0.01*i, b[i] = 0.30 + 0.02*i, c[i] = 0.30 + 0.03*i
// ⇒ mean[i] = 0.30 + 0.02*i  (chosen so the ×1000 quantised sums are
// exactly divisible by 3 for the Cairo / Noir baselines).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const MEAN_ELF: &[u8] = include_elf!("mean-program");

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

    let mut a = [0.0f64; 50];
    let mut b = [0.0f64; 50];
    let mut c = [0.0f64; 50];
    let mut result = [0.0f64; 50];
    for i in 0..50 {
        a[i] = 0.30 + 0.01 * (i as f64);
        b[i] = 0.30 + 0.02 * (i as f64);
        c[i] = 0.30 + 0.03 * (i as f64);
        result[i] = 0.30 + 0.02 * (i as f64);
    }
    for v in a.iter() { stdin.write(v); }
    for v in b.iter() { stdin.write(v); }
    for v in c.iter() { stdin.write(v); }
    for v in result.iter() { stdin.write(v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(MEAN_ELF);

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
