# pythran_matmul

Category: `needs_gadget: division`

Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json.

as_written: rejected

Running frontend
Running IR optimizations
Running backend

thread 'main' (149148) panicked at src/target/r1cs/trans.rs:387:18:
internal error: entered unreachable code: Unhandled variable operator return.0.0
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

rewrite: needs gadget

Native integer Q32 arithmetic cannot be lowered to R1CS; a custom wide fixed-point rescaling/division gadget would be required.

