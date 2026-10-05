// DS-1000 case 389 — SP1 host driver. h=6, w=12. Values are 0..72 in flat order.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const RESHAPE_ELF: &[u8] = include_elf!("reshape-3d-program");

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

    let h: u64 = 6;
    let w: u64 = 12;
    // a[i][j][k] = i*12 + j*4 + k as f64
    let mut a: [[[f64; 4]; 3]; 6] = [[[0.0; 4]; 3]; 6];
    for i in 0..6 {
        for j in 0..3 {
            for k in 0..4 {
                a[i][j][k] = (i * 12 + j * 4 + k) as f64;
            }
        }
    }

    let hu = h as usize;
    let wu = w as usize;
    let mut expected: Vec<f64> = Vec::with_capacity(hu * wu);
    for out_i in 0..hu {
        for out_j in 0..wu {
            let out_idx = out_i * wu + out_j;
            let i0 = out_idx / 36;
            let i1 = (out_idx / 12) % 3;
            let i2 = (out_idx / 4) % 3;
            let i3 = out_idx % 4;
            expected.push(a[i0 * 3 + i2][i1][i3]);
        }
    }

    for plane in a.iter() {
        for row in plane.iter() {
            for &v in row.iter() { stdin.write(&v); }
        }
    }
    stdin.write(&h);
    stdin.write(&w);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(RESHAPE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(RESHAPE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
