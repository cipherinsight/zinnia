//! Builds every guest binary for RISC Zero's RISC-V target and writes a table
//! with each case's program path and image ID.

use std::{fmt::Write, path::Path};

fn main() {
    let guests = risc0_build::embed_methods();
    let mut table = String::from(
        "/// Program path (ELF wrapped with the zkVM kernel) and image ID of every case.\n\
         pub const GUESTS: &[(&str, &str, [u32; 8])] = &[\n",
    );
    for guest in &guests {
        let id: [u32; 8] = guest.image_id.into();
        writeln!(table, "    ({:?}, {:?}, {id:?}),", guest.name, guest.path).unwrap();
    }
    table.push_str("];\n");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("guest_table.rs");
    std::fs::write(out, table).expect("write guest table");
}
