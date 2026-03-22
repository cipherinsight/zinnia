use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::types::{ScalarValue, StmtId, StringValue, Value};

macro_rules! ir_binary {
    ($($name:ident => $variant:expr),* $(,)?) => {
        $(
            pub fn $name(&mut self, a: &Value, b: &Value) -> Value {
                self.create_ir(&$variant, &[a.clone(), b.clone()])
            }
        )*
    };
}

macro_rules! ir_unary {
    ($($name:ident => $variant:expr),* $(,)?) => {
        $(
            pub fn $name(&mut self, a: &Value) -> Value {
                self.create_ir(&$variant, &[a.clone()])
            }
        )*
    };
}

macro_rules! ir_ternary {
    ($($name:ident => $variant:expr),* $(,)?) => {
        $(
            pub fn $name(&mut self, a: &Value, b: &Value, c: &Value) -> Value {
                self.create_ir(&$variant, &[a.clone(), b.clone(), c.clone()])
            }
        )*
    };
}

/// IR builder that accumulates IR statements and provides typed convenience
/// methods. Mirrors Python `IRBuilderImpl` from `builder_impl.py`.
pub struct IRBuilder {
    pub stmts: Vec<IRStatement>,
    /// Next available memory segment ID for dynamic array allocations.
    next_segment_id: u32,
    /// Next available array ID for dynamic ndarray metadata.
    next_array_id: u32,
}

impl Default for IRBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::cloned_ref_to_slice_refs)]
impl IRBuilder {
    pub fn new() -> Self {
        Self {
            stmts: Vec::new(),
            next_segment_id: 0,
            next_array_id: 0,
        }
    }

    /// Allocate a unique memory segment ID.
    pub fn alloc_segment_id(&mut self) -> u32 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Allocate a unique array metadata ID.
    pub fn alloc_array_id(&mut self) -> u32 {
        let id = self.next_array_id;
        self.next_array_id += 1;
        id
    }

    /// Ensure a value has an IR pointer. If it's a pure compile-time constant
    /// with no pointer, materialize it as an IR constant instruction.
    pub fn ensure_ptr(&mut self, val: &Value) -> Value {
        if val.ptr().is_some() {
            return val.clone();
        }
        match val {
            Value::Integer(s) => {
                if let Some(v) = s.static_val {
                    self.ir_constant_int(v)
                } else {
                    val.clone()
                }
            }
            Value::Float(s) => {
                if let Some(v) = s.static_val {
                    self.ir_constant_float(v)
                } else {
                    val.clone()
                }
            }
            Value::Boolean(s) => {
                if let Some(v) = s.static_val {
                    self.ir_constant_bool(v)
                } else {
                    val.clone()
                }
            }
            Value::None | Value::Class(_) => {
                // Materialize None/Class as constant 0 to avoid panics downstream
                self.ir_constant_int(0)
            }
            // For composites, extract first scalar element as fallback
            Value::List(data) | Value::Tuple(data) if !data.values.is_empty() => {
                self.ensure_ptr(&data.values[0])
            }
            _ => val.clone(),
        }
    }

    /// The core method: build an IR instruction, append the statement,
    /// and return the result `Value`.  Mirrors Python
    /// `IRBuilderImpl.create_ir(operator, args, dbg)`.
    pub fn create_ir(&mut self, ir: &IR, args: &[Value]) -> Value {
        // Materialize any pure constants so they have IR pointers
        let materialized: Vec<Value> = args.iter().map(|v| self.ensure_ptr(v)).collect();
        let ir_id = self.stmts.len() as StmtId;
        let (val, stmt) = build_ir(ir, ir_id, &materialized);
        self.stmts.push(stmt);
        val
    }

    pub fn export_ir_graph(self) -> IRGraph {
        IRGraph::new(self.stmts)
    }

    // ── Convenience helpers used by optimization passes ──────────────

    pub fn ir_constant_int(&mut self, value: i64) -> Value {
        self.create_ir(&IR::ConstantInt { value }, &[])
    }

    pub fn ir_constant_float(&mut self, value: f64) -> Value {
        self.create_ir(&IR::ConstantFloat { value }, &[])
    }

    pub fn ir_constant_bool(&mut self, value: bool) -> Value {
        self.create_ir(&IR::ConstantBool { value }, &[])
    }

    pub fn ir_constant_str(&mut self, value: String) -> Value {
        self.create_ir(&IR::ConstantStr { value }, &[])
    }

    // ── Macro-generated convenience methods ──────────────────────────

    // Logic
    ir_binary!(
        ir_logical_and => IR::LogicalAnd,
        ir_logical_or  => IR::LogicalOr,
    );
    ir_unary!(ir_logical_not => IR::LogicalNot);

    // Integer arithmetic
    ir_binary!(
        ir_add_i       => IR::AddI,
        ir_sub_i       => IR::SubI,
        ir_mul_i       => IR::MulI,
        ir_div_i       => IR::DivI,
        ir_floor_div_i => IR::FloorDivI,
        ir_mod_i       => IR::ModI,
        ir_pow_i       => IR::PowI,
    );
    ir_unary!(
        ir_abs_i  => IR::AbsI,
        ir_sign_i => IR::SignI,
        ir_inv_i  => IR::InvI,
    );

    // Float arithmetic
    ir_binary!(
        ir_add_f       => IR::AddF,
        ir_sub_f       => IR::SubF,
        ir_mul_f       => IR::MulF,
        ir_div_f       => IR::DivF,
        ir_floor_div_f => IR::FloorDivF,
        ir_mod_f       => IR::ModF,
        ir_pow_f       => IR::PowF,
    );
    ir_unary!(
        ir_abs_f  => IR::AbsF,
        ir_sign_f => IR::SignF,
    );

    // Comparisons
    ir_binary!(
        ir_equal_i                  => IR::EqI,
        ir_equal_f                  => IR::EqF,
        ir_not_equal_i              => IR::NeI,
        ir_not_equal_f              => IR::NeF,
        ir_less_than_i              => IR::LtI,
        ir_less_than_f              => IR::LtF,
        ir_less_than_or_equal_i     => IR::LteI,
        ir_less_than_or_equal_f     => IR::LteF,
        ir_greater_than_i           => IR::GtI,
        ir_greater_than_f           => IR::GtF,
        ir_greater_than_or_equal_i  => IR::GteI,
        ir_greater_than_or_equal_f  => IR::GteF,
        ir_equal_hash               => IR::EqHash,
    );

    // Selection (ternary: cond, true_val, false_val)
    ir_ternary!(
        ir_select_i => IR::SelectI,
        ir_select_f => IR::SelectF,
        ir_select_b => IR::SelectB,
    );

    // Casting
    ir_unary!(
        ir_int_cast   => IR::IntCast,
        ir_float_cast => IR::FloatCast,
        ir_bool_cast  => IR::BoolCast,
    );

    // String operations
    ir_binary!(ir_add_str => IR::AddStr, ir_print => IR::Print);
    ir_unary!(ir_str_i => IR::StrI, ir_str_f => IR::StrF);

    // Math functions           
    ir_unary!(
        ir_sin_f  => IR::SinF,
        ir_cos_f  => IR::CosF,
        ir_tan_f  => IR::TanF,
    );

    ir_unary!(
        ir_sinh_f => IR::SinHF,
        ir_cosh_f => IR::CosHF,
        ir_tanh_f => IR::TanHF,
        ir_sqrt_f => IR::SqrtF,
        ir_exp_f  => IR::ExpF,
        ir_log_f  => IR::LogF,
    );

    // Assert & expose
    ir_unary!(
        ir_assert          => IR::Assert,
        ir_expose_public_i => IR::ExposePublicI,
        ir_expose_public_f => IR::ExposePublicF,
    );

    // ── I/O ───────────────────────────────────────────────────────────

    pub fn ir_read_integer(&mut self, indices: Vec<u32>, is_public: bool) -> Value {
        self.create_ir(&IR::ReadInteger { indices, is_public }, &[])
    }

    pub fn ir_read_float(&mut self, indices: Vec<u32>, is_public: bool) -> Value {
        self.create_ir(&IR::ReadFloat { indices, is_public }, &[])
    }

    pub fn ir_read_hash(&mut self, indices: Vec<u32>, is_public: bool) -> Value {
        self.create_ir(&IR::ReadHash { indices, is_public }, &[])
    }

    // ── Memory ────────────────────────────────────────────────────────

    pub fn ir_allocate_memory(&mut self, segment_id: u32, size: u32, init_value: i64) -> Value {
        self.create_ir(
            &IR::AllocateMemory { segment_id, size, init_value },
            &[],
        )
    }

    pub fn ir_write_memory(&mut self, segment_id: u32, address: &Value, value: &Value) -> Value {
        self.create_ir(
            &IR::WriteMemory { segment_id },
            &[address.clone(), value.clone()],
        )
    }

    pub fn ir_read_memory(&mut self, segment_id: u32, address: &Value) -> Value {
        self.create_ir(&IR::ReadMemory { segment_id }, &[address.clone()])
    }

    // ── Dynamic NDArray ───────────────────────────────────────────────

    pub fn ir_allocate_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        dtype_name: String,
        max_length: u32,
        max_rank: u32,
    ) -> Value {
        self.create_ir(
            &IR::AllocateDynamicNDArrayMeta { array_id, dtype_name, max_length, max_rank },
            &[],
        )
    }

    pub fn ir_witness_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        args: &[Value],
    ) -> Value {
        self.create_ir(
            &IR::WitnessDynamicNDArrayMeta { array_id, max_rank },
            args,
        )
    }

    pub fn ir_dynamic_ndarray_get_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        address: &Value,
    ) -> Value {
        self.create_ir(
            &IR::DynamicNDArrayGetItem { array_id, segment_id },
            &[address.clone()],
        )
    }

    pub fn ir_dynamic_ndarray_set_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        address: &Value,
        value: &Value,
    ) -> Value {
        self.create_ir(
            &IR::DynamicNDArraySetItem { array_id, segment_id },
            &[address.clone(), value.clone()],
        )
    }

    // ── External calls ────────────────────────────────────────────────

    pub fn ir_invoke_external(
        &mut self,
        store_idx: u32,
        func_name: String,
        args: Vec<serde_json::Value>,
        kwargs: std::collections::HashMap<String, serde_json::Value>,
    ) -> Value {
        self.create_ir(
            &IR::InvokeExternal { store_idx, func_name, args, kwargs },
            &[],
        )
    }

    pub fn ir_poseidon_hash(&mut self, values: &[Value]) -> Value {
        self.create_ir(&IR::PoseidonHash, values)
    }
}

// ---------------------------------------------------------------------------
// build_ir — the Rust equivalent of AbstractIR.build_ir()
// ---------------------------------------------------------------------------

/// Build an IR statement and compute the result value.
/// This implements the `build_ir(ir_id, args, dbg)` method for all 79 IR types.
fn build_ir(ir: &IR, ir_id: StmtId, args: &[Value]) -> (Value, IRStatement) {
    match ir {
        // ── Constants ─────────────────────────────────────────────
        IR::ConstantInt { value } => (
            Value::Integer(ScalarValue::known(*value, ir_id)),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::ConstantFloat { value } => (
            Value::Float(ScalarValue::known(*value, ir_id)),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::ConstantBool { value } => (
            Value::Boolean(ScalarValue::known(*value, ir_id)),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::ConstantStr { value } => (
            Value::String(StringValue {
                val: value.clone(),
                ptr: ir_id,
            }),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),

        // ── Integer binary arithmetic ─────────────────────────────
        IR::AddI => int_binary_ir(ir, ir_id, args, |a, b| a.checked_add(b)),
        IR::SubI => int_binary_ir(ir, ir_id, args, |a, b| a.checked_sub(b)),
        IR::MulI => int_binary_ir(ir, ir_id, args, |a, b| a.checked_mul(b)),
        IR::DivI => int_binary_ir(ir, ir_id, args, |a, b| {
            if b != 0 { Some(a / b) } else { None }
        }),
        IR::FloorDivI => int_binary_ir(ir, ir_id, args, |a, b| {
            if b != 0 {
                Some(a.div_euclid(b))
            } else {
                None
            }
        }),
        IR::ModI => int_binary_ir(ir, ir_id, args, |a, b| {
            if b != 0 { Some(a % b) } else { None }
        }),
        IR::PowI => int_binary_ir(ir, ir_id, args, |a, b| {
            if b >= 0 {
                Some(a.pow(b as u32))
            } else {
                None
            }
        }),

        // ── Integer unary arithmetic ──────────────────────────────
        IR::AbsI => int_unary_ir(ir, ir_id, args, |a| Some(a.abs())),
        IR::SignI => int_unary_ir(ir, ir_id, args, |a| {
            Some(if a > 0 { 1 } else if a < 0 { -1 } else { 0 })
        }),
        IR::InvI => (
            Value::Integer(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),

        // ── Float binary arithmetic ───────────────────────────────
        IR::AddF => float_binary_ir(ir, ir_id, args, |a, b| a + b),
        IR::SubF => float_binary_ir(ir, ir_id, args, |a, b| a - b),
        IR::MulF => float_binary_ir(ir, ir_id, args, |a, b| a * b),
        IR::DivF => float_binary_ir(ir, ir_id, args, |a, b| a / b),
        IR::FloorDivF => float_binary_ir(ir, ir_id, args, |a, b| (a / b).floor()),
        IR::ModF => float_binary_ir(ir, ir_id, args, |a, b| a % b),
        IR::PowF => float_binary_ir(ir, ir_id, args, |a, b| a.powf(b)),

        // ── Float unary arithmetic ────────────────────────────────
        IR::AbsF => float_unary_ir(ir, ir_id, args, |a| a.abs()),
        IR::SignF => float_unary_ir(ir, ir_id, args, |a| {
            if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),

        // ── Integer comparisons ───────────────────────────────────
        IR::EqI => int_cmp_ir(ir, ir_id, args, |a, b| a == b),
        IR::NeI => int_cmp_ir(ir, ir_id, args, |a, b| a != b),
        IR::LtI => int_cmp_ir(ir, ir_id, args, |a, b| a < b),
        IR::LteI => int_cmp_ir(ir, ir_id, args, |a, b| a <= b),
        IR::GtI => int_cmp_ir(ir, ir_id, args, |a, b| a > b),
        IR::GteI => int_cmp_ir(ir, ir_id, args, |a, b| a >= b),

        // ── Float comparisons ─────────────────────────────────────
        IR::EqF => float_cmp_ir(ir, ir_id, args, |a, b| a == b),
        IR::NeF => float_cmp_ir(ir, ir_id, args, |a, b| a != b),
        IR::LtF => float_cmp_ir(ir, ir_id, args, |a, b| a < b),
        IR::LteF => float_cmp_ir(ir, ir_id, args, |a, b| a <= b),
        IR::GtF => float_cmp_ir(ir, ir_id, args, |a, b| a > b),
        IR::GteF => float_cmp_ir(ir, ir_id, args, |a, b| a >= b),

        // ── Math functions (float) ────────────────────────────────
        IR::SinF => float_unary_ir(ir, ir_id, args, |a| a.sin()),
        IR::SinHF => float_unary_ir(ir, ir_id, args, |a| a.sinh()),
        IR::CosF => float_unary_ir(ir, ir_id, args, |a| a.cos()),
        IR::CosHF => float_unary_ir(ir, ir_id, args, |a| a.cosh()),
        IR::TanF => float_unary_ir(ir, ir_id, args, |a| a.tan()),
        IR::TanHF => float_unary_ir(ir, ir_id, args, |a| a.tanh()),
        IR::SqrtF => float_unary_ir(ir, ir_id, args, |a| a.sqrt()),
        IR::ExpF => float_unary_ir(ir, ir_id, args, |a| a.exp()),
        IR::LogF => float_unary_ir(ir, ir_id, args, |a| a.ln()),

        // ── Logical ───────────────────────────────────────────────
        IR::LogicalAnd => {
            let la = args[0].int_val();
            let lb = args[1].int_val();
            let inferred = match (la, lb) {
                (Some(a), Some(b)) => Some(a != 0 && b != 0),
                _ => None,
            };
            (
                Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(
                    ir_id,
                    ir.clone(),
                    vec![ptr_of(&args[0]), ptr_of(&args[1])],
                    None,
                ),
            )
        }
        IR::LogicalOr => {
            let la = args[0].int_val();
            let lb = args[1].int_val();
            let inferred = match (la, lb) {
                (Some(a), Some(b)) => Some(a != 0 || b != 0),
                _ => None,
            };
            (
                Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(
                    ir_id,
                    ir.clone(),
                    vec![ptr_of(&args[0]), ptr_of(&args[1])],
                    None,
                ),
            )
        }
        IR::LogicalNot => {
            let la = args[0].int_val();
            let inferred = la.map(|a| a == 0);
            (
                Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
            )
        }

        // ── Selection ─────────────────────────────────────────────
        IR::SelectI => {
            let cond = args[0].bool_val().or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].int_val(),
                Some(false) => args[2].int_val(),
            };
            (
                Value::Integer(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(
                    ir_id,
                    ir.clone(),
                    vec![ptr_of(&args[0]), ptr_of(&args[1]), ptr_of(&args[2])],
                    None,
                ),
            )
        }
        IR::SelectF => {
            let cond = args[0].bool_val().or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].float_val(),
                Some(false) => args[2].float_val(),
            };
            (
                Value::Float(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(
                    ir_id,
                    ir.clone(),
                    vec![ptr_of(&args[0]), ptr_of(&args[1]), ptr_of(&args[2])],
                    None,
                ),
            )
        }
        IR::SelectB => {
            let cond = args[0].bool_val().or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].bool_val(),
                Some(false) => args[2].bool_val(),
            };
            (
                Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(
                    ir_id,
                    ir.clone(),
                    vec![ptr_of(&args[0]), ptr_of(&args[1]), ptr_of(&args[2])],
                    None,
                ),
            )
        }

        // ── Casting ───────────────────────────────────────────────
        IR::IntCast => {
            let inferred = args[0].float_val().map(|v| v as i64);
            (
                Value::Integer(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
            )
        }
        IR::FloatCast => {
            let inferred = args[0].int_val().map(|v| v as f64);
            (
                Value::Float(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
            )
        }
        IR::BoolCast => {
            let inferred = args[0].int_val().map(|v| v != 0);
            (
                Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
                IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
            )
        }

        // ── String operations ─────────────────────────────────────
        IR::AddStr => {
            let sa = args[0].string_val();
            let sb = args[1].string_val();
            let val_str = match (sa, sb) {
                (Some(a), Some(b)) => format!("{}{}", a, b),
                _ => String::new(),
            };
            (
                Value::String(StringValue {
                    val: val_str,
                    ptr: ir_id,
                }),
                IRStatement::new(
                    ir_id,
                    ir.clone(),
                    vec![ptr_of(&args[0]), ptr_of(&args[1])],
                    None,
                ),
            )
        }
        IR::StrI => (
            Value::String(StringValue {
                val: args[0]
                    .int_val()
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                ptr: ir_id,
            }),
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),
        IR::StrF => (
            Value::String(StringValue {
                val: args[0]
                    .float_val()
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                ptr: ir_id,
            }),
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),

        // ── I/O (fixed, no inference) ─────────────────────────────
        IR::ReadInteger { .. } => (
            Value::Integer(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::ReadFloat { .. } => (
            Value::Float(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::ReadHash { .. } => (
            Value::Integer(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::Print => (
            Value::None,
            IRStatement::new(
                ir_id,
                ir.clone(),
                vec![ptr_of(&args[0]), ptr_of(&args[1])],
                None,
            ),
        ),

        // ── Assert & expose (fixed, return None) ──────────────────
        IR::Assert => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),
        IR::ExposePublicI | IR::ExposePublicF => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),

        // ── Memory operations (fixed) ─────────────────────────────
        IR::AllocateMemory { .. } => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::WriteMemory { .. } => (
            Value::None,
            IRStatement::new(
                ir_id,
                ir.clone(),
                vec![ptr_of(&args[0]), ptr_of(&args[1])],
                None,
            ),
        ),
        IR::ReadMemory { .. } => (
            Value::Integer(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),
        IR::MemoryTraceEmit { .. } => (
            Value::None,
            IRStatement::new(
                ir_id,
                ir.clone(),
                vec![ptr_of(&args[0]), ptr_of(&args[1])],
                None,
            ),
        ),
        IR::MemoryTraceSeal => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),

        // ── Dynamic NDArray (fixed) ───────────────────────────────
        IR::AllocateDynamicNDArrayMeta { .. } => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::WitnessDynamicNDArrayMeta { .. } | IR::AssertDynamicNDArrayMeta { .. } => {
            let ptrs: Vec<StmtId> = args.iter().map(ptr_of).collect();
            (
                Value::None,
                IRStatement::new(ir_id, ir.clone(), ptrs, None),
            )
        }
        IR::DynamicNDArrayGetItem { .. } => (
            Value::Integer(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),
        IR::DynamicNDArraySetItem { .. } => (
            Value::None,
            IRStatement::new(
                ir_id,
                ir.clone(),
                vec![ptr_of(&args[0]), ptr_of(&args[1])],
                None,
            ),
        ),

        // ── External calls (fixed) ────────────────────────────────
        IR::InvokeExternal { .. } => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![], None),
        ),
        IR::ExportExternalI { .. } | IR::ExportExternalF { .. } => (
            Value::None,
            IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
        ),

        // ── Hash ──────────────────────────────────────────────────
        IR::PoseidonHash => {
            let ptrs: Vec<StmtId> = args.iter().map(ptr_of).collect();
            (
                Value::Integer(ScalarValue::new(None, Some(ir_id))),
                IRStatement::new(ir_id, ir.clone(), ptrs, None),
            )
        }
        IR::EqHash => (
            Value::Boolean(ScalarValue::new(None, Some(ir_id))),
            IRStatement::new(
                ir_id,
                ir.clone(),
                vec![ptr_of(&args[0]), ptr_of(&args[1])],
                None,
            ),
        ),
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn ptr_of(val: &Value) -> StmtId {
    val.ptr().unwrap_or_else(|| panic!("Value must have a pointer: {:?}", val))
}

fn int_binary_ir(
    ir: &IR,
    ir_id: StmtId,
    args: &[Value],
    op: impl Fn(i64, i64) -> Option<i64>,
) -> (Value, IRStatement) {
    let la = args[0].int_val();
    let lb = args[1].int_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => op(a, b),
        _ => None,
    };
    (
        Value::Integer(ScalarValue::new(inferred, Some(ir_id))),
        IRStatement::new(
            ir_id,
            ir.clone(),
            vec![ptr_of(&args[0]), ptr_of(&args[1])],
            None,
        ),
    )
}

fn int_unary_ir(
    ir: &IR,
    ir_id: StmtId,
    args: &[Value],
    op: impl Fn(i64) -> Option<i64>,
) -> (Value, IRStatement) {
    let la = args[0].int_val();
    let inferred = la.and_then(op);
    (
        Value::Integer(ScalarValue::new(inferred, Some(ir_id))),
        IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
    )
}

fn float_binary_ir(
    ir: &IR,
    ir_id: StmtId,
    args: &[Value],
    op: impl Fn(f64, f64) -> f64,
) -> (Value, IRStatement) {
    let la = args[0].float_val();
    let lb = args[1].float_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => Some(op(a, b)),
        _ => None,
    };
    (
        Value::Float(ScalarValue::new(inferred, Some(ir_id))),
        IRStatement::new(
            ir_id,
            ir.clone(),
            vec![ptr_of(&args[0]), ptr_of(&args[1])],
            None,
        ),
    )
}

fn float_unary_ir(
    ir: &IR,
    ir_id: StmtId,
    args: &[Value],
    op: impl Fn(f64) -> f64,
) -> (Value, IRStatement) {
    let la = args[0].float_val();
    let inferred = la.map(op);
    (
        Value::Float(ScalarValue::new(inferred, Some(ir_id))),
        IRStatement::new(ir_id, ir.clone(), vec![ptr_of(&args[0])], None),
    )
}

fn int_cmp_ir(
    ir: &IR,
    ir_id: StmtId,
    args: &[Value],
    op: impl Fn(i64, i64) -> bool,
) -> (Value, IRStatement) {
    let la = args[0].int_val();
    let lb = args[1].int_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => Some(op(a, b)),
        _ => None,
    };
    (
        Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
        IRStatement::new(
            ir_id,
            ir.clone(),
            vec![ptr_of(&args[0]), ptr_of(&args[1])],
            None,
        ),
    )
}

fn float_cmp_ir(
    ir: &IR,
    ir_id: StmtId,
    args: &[Value],
    op: impl Fn(f64, f64) -> bool,
) -> (Value, IRStatement) {
    let la = args[0].float_val();
    let lb = args[1].float_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => Some(op(a, b)),
        _ => None,
    };
    (
        Value::Boolean(ScalarValue::new(inferred, Some(ir_id))),
        IRStatement::new(
            ir_id,
            ir.clone(),
            vec![ptr_of(&args[0]), ptr_of(&args[1])],
            None,
        ),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_constant_int() {
        let mut b = IRBuilder::new();
        let v = b.ir_constant_int(42);
        assert_eq!(v.int_val(), Some(42));
        assert_eq!(v.ptr(), Some(0));
        assert_eq!(b.stmts.len(), 1);
    }

    #[test]
    fn test_builder_add_i() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(10);
        let c = b.ir_constant_int(20);
        let sum = b.create_ir(&IR::AddI, &[a, c]);
        assert_eq!(sum.int_val(), Some(30));
        assert_eq!(sum.ptr(), Some(2));
        assert_eq!(b.stmts.len(), 3);
    }

    #[test]
    fn test_builder_export_graph() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(5);
        let c = b.ir_constant_int(3);
        let _ = b.create_ir(&IR::AddI, &[a, c]);
        let graph = b.export_ir_graph();
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_build_ir_select() {
        let mut b = IRBuilder::new();
        let cond = b.ir_constant_bool(true);
        let tv = b.ir_constant_int(10);
        let fv = b.ir_constant_int(20);
        let result = b.create_ir(&IR::SelectI, &[cond, tv, fv]);
        assert_eq!(result.int_val(), Some(10));
    }
}
