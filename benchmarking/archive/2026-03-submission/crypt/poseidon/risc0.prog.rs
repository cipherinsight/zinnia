use risc0_zkvm::guest::env;
use light_poseidon::{Poseidon, PoseidonBytesHasher, parameters::bn254_x5};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};

// source start
fn convert_bytes_to_slices(bytes: Vec<u8>) -> Vec<[u8; 32]> {
    let mut padded_bytes = bytes;

    // Pad with zeros if not a multiple of 32
    let remainder = padded_bytes.len() % 32;
    if remainder != 0 {
        padded_bytes.extend(std::iter::repeat(0).take(32 - remainder));
    }

    // Convert into chunks of 32
    padded_bytes.chunks(32).map(|chunk| {
        let mut array = [0u8; 32];
        array.copy_from_slice(chunk);
        array
    }).collect()
}


fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut bytes: Vec<u8> = Vec::new();
    let mut sum = 0;
    for i in 0..10 {
        let x: u32 = env::read();
        for j in 0..4 {
            // Extract each byte using bitwise operations
            bytes.push((x >> (i * 8)) as u8);
        }
        sum += x;
    }

    assert_eq!(sum, 55);

    let mut poseidon: Poseidon<ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>> = Poseidon::<Fr>::new_circom(2).unwrap();
    let byte_slices = convert_bytes_to_slices(bytes).to_vec();
    let byte_slices: Vec<&[u8]> = byte_slices.iter().map(|chunk| chunk.as_slice()).collect();
    let hash = poseidon.hash_bytes_be(&byte_slices).unwrap();

    println!("{:?}", hash);

    // write public output to the journal
    // env::commit(&input);
}
