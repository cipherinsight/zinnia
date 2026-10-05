// DS-1000 case 501 — SP1 host driver.
// Sample inputs:
//   a[i][j][k] = (i*13*2 + j*2 + k + 1) / 100   (Float[13,13,2])
//   b[i][j]    = (i + j) % 2                    (Integer[13,13])
// Expected result[i][j] = a[i][j][b[i][j]].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const TAKE_ALONG_AXIS_ELF: &[u8] = include_elf!("take-along-axis-3d2-program");

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

    let args = Args::parse();
    if args.execute == args.prove {
        eprintln!("Error: specify either --execute or --prove");
        std::process::exit(1);
    }

    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    let mut a = [[[0f64; 2]; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..2 {
                a[i][j][k] = ((i * 13 * 2 + j * 2 + k + 1) as f64) / 100.0;
            }
        }
    }
    let mut b = [[0u64; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            b[i][j] = ((i + j) % 2) as u64;
        }
    }
    let mut expected = [[0f64; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            expected[i][j] = a[i][j][b[i][j] as usize];
        }
    }

    for plane in a.iter() {
        for row in plane.iter() {
            for &v in row.iter() {
                stdin.write(&v);
            }
        }
    }
    for row in b.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    for row in expected.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }

    if args.execute {
        let (_o, report) = client.execute(TAKE_ALONG_AXIS_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(TAKE_ALONG_AXIS_ELF);

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
