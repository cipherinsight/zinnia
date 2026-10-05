// DS-1000 case 0440 — SP1 host driver.
// Y[i][j][k] = ((i+1)*(j+1))² when j==k else 0. Concrete N=2, M=K=3.
// Expected X = [[1, 2], [2, 4], [3, 6]].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DIAG_SQRT_ELF: &[u8] = include_elf!("diag-sqrt-program");

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
    const K: usize = 3;
    let mut y = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K {
        y[i][j][k] = if j == k {
            let v = ((i as f64 + 1.0) * (j as f64 + 1.0)) as f64;
            v * v
        } else { 0.0 };
    } } }
    for plane in &y { for row in plane { for &v in row { stdin.write(&v); } } }
    let expect: [[f64; N]; M] = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
    for row in &expect { for &v in row { stdin.write(&v); } }
    if args.execute {
        let (_o, report) = client.execute(DIAG_SQRT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(DIAG_SQRT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
