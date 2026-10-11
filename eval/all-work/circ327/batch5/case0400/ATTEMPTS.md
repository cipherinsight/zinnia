# case0400

Category: `accepted_with_rewrite`

CirC rejects empty arrays; use a padding slot with known zero logical length.

as_written: rejected

Running frontend

thread 'main' (150702) panicked at src/front/zsharp/term.rs:1234:18:
called `Result::unwrap()` on an `Err` value: "Empty array"
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

rewrite: accepted

Compilation accepted.

- Encode the empty output by one zero padding slot; logical length is statically zero.
