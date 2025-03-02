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
use fibonacci_lib::PublicValuesStruct;
use rand::{Rng, SeedableRng};
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

/// The ELF (executable and linkable format) file for the Succinct RISC-V zkVM.
pub const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci-program");

/// The arguments for the command.
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

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let matrix = vec![
        5, 0, 3, 3, 7, 9, 3, 5, 2, 4, 7, 6, 8, 8, 1, 6, 7, 7, 8, 1, 5, 9, 8, 9, 4, 3, 0, 3, 5, 0,
        2, 3, 8, 1, 3, 3, 3, 7, 0, 1, 9, 9, 0, 4, 7, 3, 2, 7, 2, 0, 0, 4, 5, 5, 6, 8, 4, 1, 4, 9,
        8, 1, 1, 7, 9, 9, 3, 6, 7, 2, 0, 3, 5, 9, 4, 4, 6, 4, 4, 3, 4, 4, 8, 4, 3, 7, 5, 5, 0, 1,
        5, 9, 3, 0, 5, 0, 1, 2, 4, 2,
    ];
    let valid = 0;

    for i in 0..10 * 10 {
        stdin.write(&matrix[i]);
    }
    stdin.write(&valid);

    // println!("n: {}", args.n);

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

        let start = Instant::now();
        // Generate the proof
        let proof = client
            .prove(&pk, &stdin)
            .plonk()
            .run()
            .expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (ms): {:?}", duration.as_millis());

        println!("Successfully generated proof!");

        proof
            .save("proof-with-pis.bin")
            .expect("saving proof failed");

        let start = Instant::now();
        // Verify the proof.
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");
    }
}
