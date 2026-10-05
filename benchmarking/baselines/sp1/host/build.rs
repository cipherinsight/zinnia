//! Builds every guest binary for SP1's RISC-V target and writes a table that
//! maps each case to the path of its ELF.

use std::{fmt::Write, path::Path};

fn main() {
    let guest_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../guest");
    let guest_dir = guest_dir.canonicalize().expect("guest package exists");
    sp1_build::build_program(guest_dir.to_str().expect("utf-8 path"));

    let metadata = cargo_metadata::MetadataCommand::new()
        .manifest_path(guest_dir.join("Cargo.toml"))
        .exec()
        .expect("guest cargo metadata");
    let elfs = sp1_build::generate_elf_paths(&metadata, None).expect("guest ELF paths");

    let mut table = String::from("/// Guest ELF path of every case.\npub const GUEST_ELFS: &[(&str, &str)] = &[\n");
    for (case, path) in &elfs {
        writeln!(table, "    ({case:?}, {:?}),", path.as_str()).unwrap();
    }
    table.push_str("];\n");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("guest_elfs.rs");
    std::fs::write(out, table).expect("write guest table");
}
