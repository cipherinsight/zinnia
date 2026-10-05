// DS-1000 case 358 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ARGSORTSUM_ELF: &[u8] = include_elf!("argsort-sum-axis12-program");

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

    // a[0]=3 throughout (sum=27), a[1]=1 (sum=9), a[2]=2 (sum=18) -> argsort = [1, 2, 0]
    let mut a: [[[f64; 3]; 3]; 3] = [[[0.0; 3]; 3]; 3];
    for j in 0..3 { for k in 0..3 {
        a[0][j][k] = 3.0;
        a[1][j][k] = 1.0;
        a[2][j][k] = 2.0;
    } }
    let mut b: [[[i64; 3]; 3]; 3] = [[[0; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = (100 * i + 10 * j + k) as i64; } } }
    let perm = [1usize, 2, 0];
    let mut expected: [[[i64; 3]; 3]; 3] = [[[0; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { expected[i][j][k] = b[perm[i]][j][k]; } } }

    for i in 0..3 { for j in 0..3 { for k in 0..3 { stdin.write(&a[i][j][k]); } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 { stdin.write(&b[i][j][k]); } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 { stdin.write(&expected[i][j][k]); } } }

    if args.execute {
        let (_o, report) = client.execute(ARGSORTSUM_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ARGSORTSUM_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
