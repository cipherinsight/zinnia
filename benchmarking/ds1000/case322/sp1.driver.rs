// SP1 driver code
//! An end-to-end example of using the SP1 SDK to generate a proof of a program that can be executed
//! or have a core proof generated.
//!
//! You can run this script using the following command:
//! ```shell
//! RUST_LOG=info cargo run --release -- --execute
//! ```
//! or
//! ```shell
//! RUST_LOG=info cargo run --release -- --prove
//! ```

use alloy_sol_types::SolType;
use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci-program");

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long)]
    execute: bool,

    #[clap(long)]
    prove: bool,
}

fn main() {
    // Setup the logger.
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();

    // Parse the command line arguments.
    let args = Args::parse();

    if args.execute == args.prove {
        eprintln!("Error: You must specify either --execute or --prove");
        std::process::exit(1);
    }

    // Setup the prover client.
    let client = ProverClient::from_env();

    // Setup the inputs.
    let mut stdin = SP1Stdin::new();

    // a (2x2)
    for x in [
        1, 0,
        0, 2
    ] {
        let tmp: i32 = x;
        stdin.write(&tmp);
    }

    // result (2x2)
    for x in [
        0, 1,
        1, 0
    ] {
        let tmp: i32 = x;
        stdin.write(&tmp);
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        // Setup the program for proving.
        let (pk, vk) = client.setup(FIBONACCI_ELF);

        let start = Instant::now();
        // Generate the proof
        let proof = client
            .prove(&pk, &stdin)
            .run()
            .expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

        proof
        .save("proof-with-pis-stark.bin")
        .expect("saving proof failed");

        let start = Instant::now();
        // Verify the proof.
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");

        let start = Instant::now();
        // Generate the proof
        let proof = client
            .prove(&pk, &stdin)
            .plonk()
            .run()
            .expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-SNARK) (ms): {:?}", duration.as_millis());

        println!("Successfully generated proof!");

        proof
            .save("proof-with-pis.bin")
            .expect("saving proof failed");

        let start = Instant::now();
        // Verify the proof.
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-SNARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");    }
}
