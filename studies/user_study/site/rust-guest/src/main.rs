#![no_main]
risc0_zkvm::guest::entry!(main);
fn main() {
    risc0_zkvm::guest::env::commit_slice(b"[]");
}
