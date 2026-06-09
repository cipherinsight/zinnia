// DS-1000 case 398 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const IIR2_ELF: &[u8] = include_elf!("iir-2tap-program");

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

    const N: usize = 8;
    let a_arr: [f64; N] = [4.0, 2.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0];
    let a: f64 = 0.5;
    let b: f64 = 0.5;
    // B recurrence: B[0]=2, B[1]=2, B[2]=1.5, B[3]=2.25, B[4]=2.125,
    // B[5]=1.0625, B[6]=1.03125, B[7]=2.515625.
    let expected: [f64; N] = [2.0, 2.0, 1.5, 2.25, 2.125, 1.0625, 1.03125, 2.515625];

    for &v in a_arr.iter() { stdin.write(&v); }
    stdin.write(&a);
    stdin.write(&b);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(IIR2_ELF);
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
