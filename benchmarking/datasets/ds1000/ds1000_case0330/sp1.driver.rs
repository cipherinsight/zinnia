// DS-1000 case 330 — SP1 host driver. Float[8,6]^3 element-wise.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const POW_ELF: &[u8] = include_elf!("pow-elementwise-program");

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

    // a[i][j] = (i+1) + 0.1*(j+1), gives a non-trivial spread.
    let mut a: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = (i as f64 + 1.0) + 0.1 * (j as f64 + 1.0);
        }
    }
    let power: u64 = 3;

    for row in a.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    stdin.write(&power);

    // Pre-compute claimed = a^3 on the host.
    let mut claimed: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            let mut acc: f64 = 1.0;
            for _ in 0..power {
                acc *= a[i][j];
            }
            claimed[i][j] = acc;
        }
    }
    for row in claimed.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(POW_ELF);

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
