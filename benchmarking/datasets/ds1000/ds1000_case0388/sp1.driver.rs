// DS-1000 case 388 — SP1 host driver. patch_size = 4; H/4*4 = 100, W/4*4 = 200.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PATCH_ELF: &[u8] = include_elf!("patch-param-program");

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

    const H: usize = 100;
    const W: usize = 200;
    let p: u64 = 4;
    let pu: usize = p as usize;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for i in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for j in 0..W { row.push(i as f64 * 0.001 + j as f64 * 0.0001); }
        a.push(row);
    }

    let h_eff = (H / pu) * pu;
    let w_eff = (W / pu) * pu;
    let r_blocks = h_eff / pu;
    let c_blocks = w_eff / pu;
    let mut expected: Vec<f64> = Vec::with_capacity(r_blocks * c_blocks * pu * pu);
    for r_block in 0..r_blocks {
        for c_block in 0..c_blocks {
            for pp in 0..pu {
                for qq in 0..pu {
                    expected.push(a[r_block * pu + pp][c_block * pu + qq]);
                }
            }
        }
    }

    for row in a.iter() { for &v in row.iter() { stdin.write(&v); } }
    stdin.write(&p);
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(PATCH_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PATCH_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
