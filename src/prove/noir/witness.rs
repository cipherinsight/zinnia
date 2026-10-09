//! The same constraint-free witness transport as benchmarking/noir_witness.py.
//! bb ignores unused private ABI parameters when finding its highest witness
//! index. With no constraints/publics it expects one unused slot. Never publish
//! an input, add a constraint, or discard a witness of a nonempty circuit.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use base64::Engine;
use serde::Deserialize;
use serde::de::IgnoredAny;
use super::driver::NoirError;

#[derive(Deserialize)]
struct Program(Vec<Circuit>, Vec<IgnoredAny>);
#[derive(Deserialize)]
struct Circuit(String, u32, Vec<IgnoredAny>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<IgnoredAny>);

fn constraint_free(raw: &[u8]) -> bool {
    if raw.first() != Some(&3) { return false; }
    let Ok(Program(functions, unconstrained)) = rmp_serde::from_slice::<Program>(&raw[1..]) else { return false; };
    if functions.len() != 1 || !unconstrained.is_empty() { return false; }
    let Circuit(_name, _highest, ops, _private, public, returns, _messages) = &functions[0];
    ops.is_empty() && public.is_empty() && returns.is_empty()
}

pub(super) fn for_barretenberg(bytecode: &Path, witness: &Path) -> Result<PathBuf, NoirError> {
    let adapt = || -> Result<PathBuf, Box<dyn std::error::Error>> {
        let artifact: serde_json::Value = serde_json::from_slice(&std::fs::read(bytecode)?)?;
        let encoded = artifact["bytecode"].as_str().ok_or("missing bytecode")?;
        let compressed = base64::engine::general_purpose::STANDARD.decode(encoded)?;
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(compressed.as_slice()).read_to_end(&mut raw)?;
        if !constraint_free(&raw) { return Ok(witness.to_path_buf()); }
        let mut transport = vec![3, 0x91, 0x91, 0x92, 0, 0x81, 0, 0xc4, 0x20];
        transport.extend([0; 32]);
        let target = witness.with_extension("bb.gz");
        let mut gzip = flate2::write::GzEncoder::new(std::fs::File::create(&target)?, flate2::Compression::default());
        gzip.write_all(&transport)?;
        gzip.finish()?;
        Ok(target)
    };
    adapt().map_err(|e| NoirError::Io(format!("constraint-free witness transport: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_constraint_free_single_function_without_publics_is_adapted() {
        let encode = |ops: Vec<u32>, public: Vec<u32>, returns: Vec<u32>| {
            let mut bytes = vec![3];
            bytes.extend(rmp_serde::to_vec(&(vec![("main", 99u32, ops, vec![0u32, 99], public, returns, Vec::<u32>::new())], Vec::<u32>::new())).unwrap());
            bytes
        };
        assert!(constraint_free(&encode(vec![], vec![], vec![])));
        assert!(!constraint_free(&encode(vec![1], vec![], vec![])));
        assert!(!constraint_free(&encode(vec![], vec![0], vec![])));
        assert!(!constraint_free(&encode(vec![], vec![], vec![0])));
        assert!(!constraint_free(&[3, 0x90]));
    }
}
