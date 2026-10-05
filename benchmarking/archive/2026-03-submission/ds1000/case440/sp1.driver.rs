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

    let y: [[[f32; 3]; 3]; 4] = [
        [[81.0, 63.0, 63.0], [63.0, 49.0, 49.0], [63.0, 49.0, 49.0]],
        [[4.0, 12.0, 8.0], [12.0, 36.0, 24.0], [8.0, 24.0, 16.0]],
        [[25.0, 35.0, 25.0], [35.0, 49.0, 35.0], [25.0, 35.0, 25.0]],
        [[25.0, 30.0, 10.0], [30.0, 36.0, 12.0], [10.0, 12.0, 4.0]],
    ];
    for i in 0..4 {
        for j in 0..3 {
            for k in 0..3 {
                stdin.write(&y[i][j][k]);
            }
        }
    }

    let mut x: [[f32; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..4 {
        for j in 0..3 {
            x[j][i] = y[i][j][j].sqrt();
        }
    }

    for i in 0..3 {
        for j in 0..4 {
            stdin.write(&x[i][j]);
        }
    }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(FIBONACCI_ELF);

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin)
            .run()
            .expect("failed to generate proof");
        let duration = start.elapsed();
        println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

        proof.save("proof-with-pis-stark.bin").expect("saving proof failed");

        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        let duration = start.elapsed();
        println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
        println!("Successfully verified proof!");

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin)
            .plonk()
            .run()
            .expect("failed to generate proof");
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
