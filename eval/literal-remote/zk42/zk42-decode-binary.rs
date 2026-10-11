use std::{env,fs,path::Path};
use cairo_air::utils::{deserialize_proof_from_file,ProofFormat};
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args:Vec<String>=env::args().collect();
    if args.len()!=3 { return Err("usage: zk42-decode-binary PROOF.bin OUTPUT.json".into()); }
    let proof=deserialize_proof_from_file::<Blake2sMerkleHasherGeneric<false>>(Path::new(&args[1]),ProofFormat::Binary)?;
    fs::write(&args[2],serde_json::to_vec(&proof)?)?;
    Ok(())
}
