// DS-1000 case 374 — SP1 host driver.
// grades = 1..=50 (already sorted ascending); eval = [0, 10, 25, 50, 75].
// xs = grades; ys[i] = (i+1)/50; expected = [0, 10/50, 25/50, 1, 1].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ECDFINT_ELF: &[u8] = include_elf!("ecdf-interp-program");

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

    let mut grades: [f64; 50] = [0.0; 50];
    for i in 0..50 { grades[i] = (i + 1) as f64; }
    let eval: [i64; 5] = [0, 10, 25, 50, 75];
    // xs sorted == grades. ys[k] = (k+1)/50.
    // element=0 < xs[0]=1 → 0
    // element=10 >= xs[0], < xs[49]=50. argmax(xs>10) = first k with xs[k]>10 = k=10 (xs[10]=11).
    //   result = ys[9] = 10/50 = 0.2
    // element=25: first k with xs[k]>25 = 25 (xs[25]=26). result = ys[24] = 25/50 = 0.5
    // element=50 >= xs[49]=50 → 1
    // element=75 >= xs[49]=50 → 1
    let expected: [f64; 5] = [0.0, 0.2, 0.5, 1.0, 1.0];

    for &v in grades.iter() { stdin.write(&v); }
    for &v in eval.iter() { stdin.write(&v); }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(ECDFINT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ECDFINT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
