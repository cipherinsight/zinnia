// DS-1000 case 403 — SP1 host driver. ravel_multi_index order='C'.
// dims=[8,9,8,8,8]; index=[2,3,4,5,6]; expected = 11054.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const RAVEL_C_ELF: &[u8] = include_elf!("ravel-multi-index-c-program");

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

    let dims: [u64; 5] = [8, 9, 8, 8, 8];
    let index: [u64; 5] = [2, 3, 4, 5, 6];
    let expected: u64 = 11054;

    for &d in dims.iter() { stdin.write(&d); }
    let total: usize = 8 * 9 * 8 * 8 * 8;
    for k in 0..total {
        let v: f64 = (k as f64) * 0.001;
        stdin.write(&v);
    }
    for &i in index.iter() { stdin.write(&i); }
    stdin.write(&expected);

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(RAVEL_C_ELF);
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
