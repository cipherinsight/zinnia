// DS-1000 case 0498 — SP1 host driver. N=2, M=3, K=2, target shape=(4, 5).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PAD_3D_ELF: &[u8] = include_elf!("pad-3d-program");

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args { #[clap(long)] execute: bool, #[clap(long)] prove: bool, }

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let args = Args::parse();
    if args.execute == args.prove { eprintln!("Error"); std::process::exit(1); }
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 2;
    let new_n: u32 = 4;
    let new_m: u32 = 5;
    let mut arr = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K {
        arr[i][j][k] = (i as f64) * 100.0 + (j as f64) * 10.0 + (k as f64);
    } } }
    stdin.write(&new_n);
    stdin.write(&new_m);
    for plane in &arr { for row in plane { for &v in row { stdin.write(&v); } } }
    for i in 0..new_n as usize {
        for j in 0..new_m as usize {
            for k in 0..K {
                let v: f64 = if i < N && j < M { arr[i][j][k] } else { 0.0 };
                stdin.write(&v);
            }
        }
    }
    if args.execute {
        let (_o, report) = client.execute(PAD_3D_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PAD_3D_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
