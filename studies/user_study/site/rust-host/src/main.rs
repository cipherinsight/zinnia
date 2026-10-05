use risc0_zkvm::{default_executor, ExecutorEnv};
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("guest ELF path required")?;
    let elf = std::fs::read(path)?;
    let program = risc0_binfmt::ProgramBinary::new(&elf, risc0_zkos_v1compat::V1COMPAT_ELF).encode();
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;
    let env = ExecutorEnv::builder()
        .write_slice(&input)
        .session_limit(Some(32 * 1024 * 1024))
        .build()?;
    let session = default_executor().execute(env, &program)?;
    std::fs::write("result.json", &session.journal.bytes)?;
    Ok(())
}
