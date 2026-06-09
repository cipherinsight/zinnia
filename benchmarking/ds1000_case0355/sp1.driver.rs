// DS-1000 case 355 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ARGSORTAXIS0_ELF: &[u8] = include_elf!("argsort-axis0-gather-program");

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

    // a chosen so argsort axis 0 produces a known permutation per (j,k).
    // For every (j,k), make a[0]=3.0, a[1]=1.0, a[2]=2.0 -> argsort = [1,2,0].
    let a: [[[f64; 3]; 3]; 3] = [
        [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
    ];
    // b distinct values so reorder is observable: b[i][j][k] = 100*i + 10*j + k.
    let mut b: [[[i64; 3]; 3]; 3] = [[[0; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = (100 * i + 10 * j + k) as i64; } } }
    // c[i][j][k] = b[sort_idx[i][j][k]][j][k] with sort_idx along i = [1,2,0]
    let mut expected: [[[i64; 3]; 3]; 3] = [[[0; 3]; 3]; 3];
    let perm = [1usize, 2, 0];
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        expected[i][j][k] = b[perm[i]][j][k];
    } } }

    for i in 0..3 { for j in 0..3 { for k in 0..3 { stdin.write(&a[i][j][k]); } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 { stdin.write(&b[i][j][k]); } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 { stdin.write(&expected[i][j][k]); } } }

    if args.execute {
        let (_o, report) = client.execute(ARGSORTAXIS0_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ARGSORTAXIS0_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
