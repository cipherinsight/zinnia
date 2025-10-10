// The ELF is used for proving and the ID is used for verification.
use alloy_sol_types::SolType;
use clap::Parser;
use fibonacci_lib::PublicValuesStruct;
use rand::{Rng, SeedableRng};
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;
use ethereum_types::U512;

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

    let leaves = [
        11u64, 22, 33, 44, 55, 66, 77, 88
    ].map(U512::from);
    let leaf_idx = U512::from(5u64);
    let path = [
        U512::from(55u64),
        U512::from_str_radix("17601130510839997314447930637874687829291919088912867269545606748119523760824", 10).unwrap(),
        U512::from_str_radix("16113907914567631108494203727747775918365688705733099625555899303940350166804", 10).unwrap(),
    ];
    let bits = [U512::from(1u64), U512::from(0u64), U512::from(1u64)];

    for v in &leaves { stdin.write(v); }
    stdin.write(&leaf_idx);
    for v in &path { stdin.write(v); }
    for v in &bits { stdin.write(v); }

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
        println!("Successfully verified proof!");
    }
}
