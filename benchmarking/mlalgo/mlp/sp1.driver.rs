// SP1 driver code
use alloy_sol_types::SolType;
use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci-program");

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
        eprintln!("Error: You must specify either --execute or --prove");
        std::process::exit(1);
    }

    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    // Inputs
    let training_x: [[f64; 2]; 10] = [
        [-2.0, -1.0],
        [-2.0,  1.0],
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 0.0,  0.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [ 2.0, -1.0],
        [ 2.0,  1.0],
        [ 2.0,  0.0],
    ];
    let mut training_y: [f64; 10] = [0.0; 10];
    for i in 0..10 {
        let x1 = training_x[i][0];
        let x2 = training_x[i][1];
        let t = x1 + 0.5 * x2 + 1.0;
        training_y[i] = t * t + 0.5;
    }
    let testing_x: [[f64; 2]; 2] = [
        [-1.0, 0.0],
        [ 1.0, 2.0],
    ];
    let mut testing_y: [f64; 2] = [0.0; 2];
    for i in 0..2 {
        let x1 = testing_x[i][0];
        let x2 = testing_x[i][1];
        let t = x1 + 0.5 * x2 + 1.0;
        testing_y[i] = t * t + 0.5;
    }

    for i in 0..10 {
        stdin.write(&training_x[i][0]);
        stdin.write(&training_x[i][1]);
    }
    for i in 0..10 {
        stdin.write(&training_y[i]);
    }
    for i in 0..2 {
        stdin.write(&testing_x[i][0]);
        stdin.write(&testing_x[i][1]);
    }
    for i in 0..2 {
        stdin.write(&testing_y[i]);
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(FIBONACCI_ELF);

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).run().expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

        proof.save("proof-with-pis-stark.bin").expect("saving proof failed");

        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).plonk().run().expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-SNARK) (ms): {:?}", duration.as_millis());

        println!("Successfully generated proof!");

        proof.save("proof-with-pis.bin").expect("saving proof failed");

        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-SNARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");
    }
}
