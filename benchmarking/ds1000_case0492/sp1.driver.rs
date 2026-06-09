// DS-1000 case 492 — SP1 host driver.
// s1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; s2 = [1.0, 2.5, 3.0, 4.1, 5.0, 6.000001].
// |diff|: [0, 0.5, 0, 0.1, 0, 1e-6]; bound ≈ atol + rtol*|s2|.
// Not-close: indices 1, 3. Expected count = 2.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ISCLOSE_EQNAN_ELF: &[u8] = include_elf!("isclose-eqnan-program");

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

    let s1: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let s2: [f64; 6] = [1.0, 2.5, 3.0, 4.1, 5.0, 6.000001];
    let claimed: i64 = 2;
    for &v in s1.iter() { stdin.write(&v); }
    for &v in s2.iter() { stdin.write(&v); }
    stdin.write(&claimed);

    if args.execute {
        let (_o, report) = client.execute(ISCLOSE_EQNAN_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ISCLOSE_EQNAN_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis-stark.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).plonk().run().unwrap();
    println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
}
