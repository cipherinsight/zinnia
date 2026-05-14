//! Tests for the structured-event telemetry sink (Phase 1 of the
//! verification-loop epic; card `compiler.verification-telemetry-sink`).
//!
//! Coverage:
//!
//! 1. `telemetry_emits_discharge_event` — env var set ⇒ a discharge fires
//!    on `sqrt_f` (with `x >= 0` planted) ⇒ the resulting `.jsonl` file
//!    contains the discharge event.
//! 2. `telemetry_no_op_when_env_unset` — env var unset ⇒ no file created,
//!    no panic, sink field is `None`.
//! 3. `telemetry_emits_strategy_dispatch_event` — env var set, build a
//!    minimal `OpStrategySet`, plant the gated precondition, dispatch ⇒
//!    `strategy_dispatch` line written.
//! 4. `telemetry_records_compile_summary` — env var set, emit
//!    `CompileStart` / `CompileEnd` markers explicitly around a tiny
//!    build, assert both events appear with non-zero `ir_count`.
//!
//! Env-var tests share a `TELEMETRY_ENV_LOCK` so parallel test execution
//! doesn't race on `ZINNIA_TELEMETRY_DIR`. Pattern mirrors `STRICT_ENV_LOCK`
//! in `helpers/array_ops/indexing.rs`.

#[cfg(test)]
mod tests {
    use crate::builder::IRBuilder;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::{
        dispatch_strategy, CostHint, OpStrategy, OpStrategySet, TelemetryEvent, TelemetrySink,
    };
    use crate::types::ValueId;
    use std::io::Read;
    use std::sync::Mutex;

    /// Serialises env-var-mutating telemetry tests. `cargo test` runs in
    /// parallel by default; without this, one test's `remove_var` would
    /// race another's `set_var`. Same shape as `STRICT_ENV_LOCK` in
    /// `helpers/array_ops/indexing.rs`.
    static TELEMETRY_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that sets `ZINNIA_TELEMETRY_DIR` to a fresh tmp dir for
    /// the lifetime of the guard, and removes it on drop. Holds the
    /// `TELEMETRY_ENV_LOCK` to serialise concurrent tests.
    struct ScopedTelemetryDir {
        dir: std::path::PathBuf,
        previous: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl ScopedTelemetryDir {
        fn enabled() -> Self {
            let lock = TELEMETRY_ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var("ZINNIA_TELEMETRY_DIR").ok();
            // Unique-per-test tmp dir to keep file-listing assertions simple.
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join(format!(
                "zinnia-tel-test-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("tmp dir creatable");
            std::env::set_var("ZINNIA_TELEMETRY_DIR", &dir);
            Self {
                dir,
                previous,
                _lock: lock,
            }
        }

        /// Acquire the lock and ensure the env var is unset for this test.
        fn disabled() -> Self {
            let lock = TELEMETRY_ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var("ZINNIA_TELEMETRY_DIR").ok();
            std::env::remove_var("ZINNIA_TELEMETRY_DIR");
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join(format!(
                "zinnia-tel-test-disabled-{}-{nonce}",
                std::process::id()
            ));
            // No mkdir — we want the test to confirm the dir is NOT touched.
            Self {
                dir,
                previous,
                _lock: lock,
            }
        }

        fn dir(&self) -> &std::path::Path {
            &self.dir
        }
    }

    impl Drop for ScopedTelemetryDir {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var("ZINNIA_TELEMETRY_DIR", v),
                None => std::env::remove_var("ZINNIA_TELEMETRY_DIR"),
            }
            // Best-effort cleanup of the tmp dir. Failures are ignored —
            // /tmp gets purged anyway.
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Read every `.jsonl` line written by the sink at `path`, joined.
    /// Returns an empty string if the path doesn't exist (lets the
    /// "no file" test assert cheaply via emptiness).
    fn read_sink_file(path: &std::path::Path) -> String {
        let mut s = String::new();
        if let Ok(mut f) = std::fs::File::open(path) {
            let _ = f.read_to_string(&mut s);
        }
        s
    }

    /// Collect every regular file in `dir` (non-recursive). Empty if the
    /// directory doesn't exist.
    fn list_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                if entry.path().is_file() {
                    out.push(entry.path());
                }
            }
        }
        out
    }

    /// Plant `vid >= k` on `vid`.
    fn plant_ge(b: &mut IRBuilder, vid: ValueId, k: i64) {
        let fact = ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(k)),
        };
        b.facts.insert_for(vid, fact);
    }

    /// Plant `Var(Value(vid)) >= 0.0` (float form) on `vid`. Needed to
    /// drive `sqrt_f`'s `Var(Formal("x")) >= 0.0` requires to `Proved`.
    fn plant_ge_zero_float(b: &mut IRBuilder, vid: ValueId) {
        let fact = ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            )),
        };
        b.facts.insert_for(vid, fact);
    }

    #[test]
    fn telemetry_emits_discharge_event() {
        let env = ScopedTelemetryDir::enabled();
        let mut b = IRBuilder::new();
        assert!(
            b.telemetry().is_some(),
            "sink should auto-attach when ZINNIA_TELEMETRY_DIR is set"
        );
        let sink_path = b.telemetry().unwrap().path.clone();

        // Build a sqrt_f input wire with a planted `>= 0.0` fact so the
        // op's `requires(x >= 0.0)` discharges as Proved (rather than
        // emitting a witness check, which would also be a valid event but
        // adds IR-emission noise to the test).
        let x = b.ir_constant_float(4.0);
        let x_vid = x.value_id().expect("constant_float carries a value_id");
        plant_ge_zero_float(&mut b, x_vid);
        let _y = b.ir_sqrt_f(&x);

        // Drop b so any buffered I/O flushes on Mutex drop.
        drop(b);

        let contents = read_sink_file(&sink_path);
        assert!(
            contents.contains(r#""event":"discharge""#),
            "expected a discharge event in {sink_path:?}; got:\n{contents}"
        );
        assert!(
            contents.contains(r#""op":"sqrt_f""#),
            "discharge event should name op=sqrt_f; got:\n{contents}"
        );
        assert!(
            contents.contains(r#""outcome":"Proved""#),
            "planting `x >= 0.0` should make sqrt_f's requires Proved; got:\n{contents}"
        );
        // sqrt_f also has an ensures clause → expect at least one
        // fact_emit on the same line.
        assert!(
            contents.contains(r#""event":"fact_emit""#),
            "sqrt_f's ensures should produce a fact_emit event; got:\n{contents}"
        );
        // Keep `env` alive until end of test for the dir cleanup.
        drop(env);
    }

    #[test]
    fn telemetry_no_op_when_env_unset() {
        let env = ScopedTelemetryDir::disabled();
        let mut b = IRBuilder::new();
        assert!(
            b.telemetry().is_none(),
            "sink should be None when ZINNIA_TELEMETRY_DIR is unset"
        );

        // Same compile path as the enabled test. Should be a complete
        // no-op for telemetry.
        let x = b.ir_constant_float(4.0);
        let x_vid = x.value_id().expect("constant_float carries a value_id");
        plant_ge_zero_float(&mut b, x_vid);
        let _y = b.ir_sqrt_f(&x);
        drop(b);

        let files = list_files(env.dir());
        assert!(
            files.is_empty(),
            "no telemetry files should exist when env is unset; found {files:?}"
        );
    }

    #[test]
    fn telemetry_emits_strategy_dispatch_event() {
        let env = ScopedTelemetryDir::enabled();
        let mut b = IRBuilder::new();
        let sink_path = b.telemetry().unwrap().path.clone();

        // Plant `vid >= 0` so the gated strategy's precondition is Proved.
        let vid = ValueId::next();
        plant_ge(&mut b, vid, 0);

        fn gated_lower(_b: &mut IRBuilder, _inputs: &ValueId) -> &'static str {
            "gated"
        }
        fn default_lower(_b: &mut IRBuilder, _inputs: &ValueId) -> &'static str {
            "default"
        }

        let set: OpStrategySet<ValueId, &'static str> = OpStrategySet {
            strategies: vec![OpStrategy {
                name: "ge_zero_fast",
                precondition: ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
                    rhs: Box::new(ContractTerm::LitInt(0)),
                },
                cost_hint: CostHint::O1,
                lower: gated_lower,
            }],
            default: default_lower,
        };
        let out = dispatch_strategy(&mut b, "telemetry_test_op", &vid, &set);
        assert_eq!(out, "gated", "gated strategy should fire on Proved");

        drop(b);
        let contents = read_sink_file(&sink_path);
        assert!(
            contents.contains(r#""event":"strategy_dispatch""#),
            "expected strategy_dispatch event; got:\n{contents}"
        );
        assert!(
            contents.contains(r#""op":"telemetry_test_op""#),
            "strategy_dispatch must carry op name; got:\n{contents}"
        );
        assert!(
            contents.contains(r#""strategy":"ge_zero_fast""#),
            "strategy_dispatch must carry strategy name; got:\n{contents}"
        );
        // The precondition references `vid` — its u64 representation
        // should appear in `input_value_ids`.
        let needle = format!(r#""input_value_ids":[{}]"#, vid.0);
        assert!(
            contents.contains(&needle),
            "strategy_dispatch should serialize input_value_ids correctly (expected {needle}); got:\n{contents}"
        );
        drop(env);
    }

    #[test]
    fn telemetry_records_compile_summary() {
        let env = ScopedTelemetryDir::enabled();
        let mut b = IRBuilder::new();
        let sink_path = b.telemetry().unwrap().path.clone();

        // The CompileStart/CompileEnd events aren't auto-emitted yet (the
        // IR-gen entrypoint wiring is a follow-up — this card just ships
        // the sink + 3 in-builder sites). For now, callers that want
        // compile-boundary markers emit them explicitly. We exercise that
        // surface here: emit a Start, build a couple of IR ops, emit an
        // End with the resulting `ir_count`.
        let t0 = std::time::Instant::now();
        if let Some(sink) = b.telemetry() {
            sink.emit(&TelemetryEvent::CompileStart {
                program: "telemetry_test_program".into(),
                timestamp_ns: crate::optim::telemetry::now_unix_ns(),
            });
        }

        // Tiny build — two int constants is enough to register non-zero
        // ir_count.
        let _a = b.ir_constant_int(1);
        let _bv = b.ir_constant_int(2);
        let ir_count = b.stmts.len();
        let duration_ms = t0.elapsed().as_millis() as u64;
        if let Some(sink) = b.telemetry() {
            sink.emit(&TelemetryEvent::CompileEnd {
                program: "telemetry_test_program".into(),
                ir_count,
                duration_ms,
            });
        }
        drop(b);

        let contents = read_sink_file(&sink_path);
        assert!(
            contents.contains(r#""event":"compile_start""#),
            "expected compile_start event; got:\n{contents}"
        );
        assert!(
            contents.contains(r#""event":"compile_end""#),
            "expected compile_end event; got:\n{contents}"
        );
        assert!(
            contents.contains(r#""program":"telemetry_test_program""#),
            "compile events should carry the program name; got:\n{contents}"
        );
        // ir_count must be non-zero (we built 2 constants).
        assert!(
            contents.contains(r#""ir_count":2"#),
            "compile_end should record ir_count=2; got:\n{contents}"
        );
        drop(env);
    }

    /// Sanity check that `TelemetrySink::from_env()` is `None` for an
    /// empty env-var value (treated as "unset" rather than "use current
    /// directory" — defensive against shell-script `export VAR=` patterns).
    #[test]
    fn telemetry_treats_empty_env_var_as_disabled() {
        let _lock = TELEMETRY_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("ZINNIA_TELEMETRY_DIR").ok();
        std::env::set_var("ZINNIA_TELEMETRY_DIR", "");
        let sink = TelemetrySink::from_env();
        match previous {
            Some(v) => std::env::set_var("ZINNIA_TELEMETRY_DIR", v),
            None => std::env::remove_var("ZINNIA_TELEMETRY_DIR"),
        }
        assert!(sink.is_none(), "empty env var should yield no sink");
    }

}
