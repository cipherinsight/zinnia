// DS-1000 case 318 — SP1 host driver.
// permutation = [2, 0, 4, 1, 5, 3] → inverse c = [1, 3, 0, 5, 2, 4].
// a[i][j] = (i+1)*10 + (j+1) → out[i][j] = a[i][c[j]].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const INV_PERM_COL_ELF: &[u8] = include_elf!("inverse-permute-cols-program");

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

    let mut a: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = ((i + 1) * 10 + (j + 1)) as f64;
        }
    }
    let permutation: [i64; 6] = [2, 0, 4, 1, 5, 3];
    let c: [i64; 6] = [1, 3, 0, 5, 2, 4];

    for row in a.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }
    for &p in permutation.iter() {
        stdin.write(&p);
    }

    let mut out: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            out[i][j] = a[i][c[j] as usize];
        }
    }
    for row in out.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }

    if args.execute {
        let (_o, report) = client.execute(INV_PERM_COL_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(INV_PERM_COL_ELF);
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
