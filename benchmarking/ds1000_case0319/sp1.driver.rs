// DS-1000 case 319 — SP1 host driver.
// permutation = [2, 0, 4, 1, 5, 3, 7, 6] → c = [1, 3, 0, 5, 2, 4, 7, 6].
// a[i][j][k] = 100*(i+1) + 10*(j+1) + (k+1) (small, distinct).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const INV_PERM_ROWS_ELF: &[u8] = include_elf!("inverse-permute-rows-3d-program");

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

    let mut a: [[[f64; 8]; 6]; 8] = [[[0.0; 8]; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            for k in 0..8 {
                a[i][j][k] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
            }
        }
    }
    let permutation: [i64; 8] = [2, 0, 4, 1, 5, 3, 7, 6];
    let c: [i64; 8] = [1, 3, 0, 5, 2, 4, 7, 6];

    for plane in a.iter() {
        for row in plane.iter() {
            for &v in row.iter() {
                stdin.write(&v);
            }
        }
    }
    for &p in permutation.iter() {
        stdin.write(&p);
    }

    // out[i][j][k] = a[c[i]][j][k].
    for i in 0..8 {
        let src_i: usize = c[i] as usize;
        for j in 0..6 {
            for k in 0..8 {
                let v: f64 = a[src_i][j][k];
                stdin.write(&v);
            }
        }
    }

    if args.execute {
        let (_o, report) = client.execute(INV_PERM_ROWS_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(INV_PERM_ROWS_ELF);
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
