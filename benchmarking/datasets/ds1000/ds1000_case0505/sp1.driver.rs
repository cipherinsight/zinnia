// DS-1000 case 505 — SP1 host driver.
// Sample inputs:
//   a[i][j][k] = (i*65 + j*5 + k + 1) / 100   (Float[13,13,5])
//   b[i][j]    = (i + j) % 5
// expected = sum(a) - sum(a[i,j,b[i,j]]).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const TAKE_ALONG_AXIS_COMPLEMENT_ELF: &[u8] = include_elf!("take-along-axis-complement-program");

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

    let mut a = [[[0f64; 5]; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                a[i][j][k] = ((i * 13 * 5 + j * 5 + k + 1) as f64) / 100.0;
            }
        }
    }
    let mut b = [[0u64; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            b[i][j] = ((i + j) % 5) as u64;
        }
    }
    let mut sum_a: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                sum_a += a[i][j][k];
            }
        }
    }
    let mut sum_arr: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            sum_arr += a[i][j][b[i][j] as usize];
        }
    }
    let expected: f64 = sum_a - sum_arr;

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
    stdin.write(&expected);

    if args.execute {
        let (_o, report) = client.execute(TAKE_ALONG_AXIS_COMPLEMENT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }

    let (pk, vk) = client.setup(TAKE_ALONG_AXIS_COMPLEMENT_ELF);

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
