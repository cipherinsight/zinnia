//! Mock execution engine for Zinnia IR.
//!
//! Evaluates IR statements with concrete values using BN254 prime field
//! arithmetic. Used for development-time testing ("mock proving") without
//! generating actual ZK proofs.

use std::collections::HashMap;
use std::str::FromStr;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::ir::IRGraph;
use crate::ir_defs::{ExternalKey, IR};

// ---------------------------------------------------------------------------
// BN254 field arithmetic
// ---------------------------------------------------------------------------

/// The BN254 scalar field prime (Fr order).
fn bn254_prime() -> BigInt {
    BigInt::from_str(
        "21888242871839275222246405745257275088548364400416034343698204186575808495617",
    )
    .unwrap()
}

fn bn254_half_prime() -> BigInt {
    &bn254_prime() >> 1
}

/// Reduce v into [0, p).
fn to_field(v: &BigInt) -> BigInt {
    v.mod_floor(&bn254_prime())
}

/// Interpret a field element as a signed integer in [-(p-1)/2, (p-1)/2].
fn from_signed(v: &BigInt) -> BigInt {
    if *v > bn254_half_prime() {
        v - bn254_prime()
    } else {
        v.clone()
    }
}

fn field_add(a: &BigInt, b: &BigInt) -> BigInt {
    to_field(&(a + b))
}

fn field_sub(a: &BigInt, b: &BigInt) -> BigInt {
    to_field(&(a - b + bn254_prime()))
}

fn field_mul(a: &BigInt, b: &BigInt) -> BigInt {
    to_field(&(a * b))
}

fn field_inv(v: &BigInt) -> BigInt {
    if v.is_zero() {
        BigInt::zero()
    } else {
        v.modpow(&(bn254_prime() - BigInt::from(2)), &bn254_prime())
    }
}

fn field_div(a: &BigInt, b: &BigInt) -> BigInt {
    field_mul(a, &field_inv(b))
}

fn field_pow(base: &BigInt, exp: &BigInt) -> BigInt {
    let signed_exp = from_signed(exp);
    if signed_exp < BigInt::zero() {
        // Negative exponent: base^(-exp) = inv(base^exp)
        let pos = (-signed_exp).to_biguint().unwrap();
        let result = base
            .to_biguint()
            .unwrap()
            .modpow(&pos, &bn254_prime().to_biguint().unwrap());
        field_inv(&BigInt::from(result))
    } else {
        let pos = signed_exp.to_biguint().unwrap();
        let result = base
            .to_biguint()
            .unwrap()
            .modpow(&pos, &bn254_prime().to_biguint().unwrap());
        BigInt::from(result)
    }
}

fn bool_to_field(b: bool) -> BigInt {
    if b {
        BigInt::one()
    } else {
        BigInt::zero()
    }
}

fn is_truthy(v: &BigInt) -> bool {
    !v.is_zero()
}

// ---------------------------------------------------------------------------
// Mock value types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum MockValue {
    Integer(BigInt),
    Float(f64),
    Boolean(bool),
    Str(String),
    None,
}

impl MockValue {
    fn as_int(&self) -> Result<&BigInt, String> {
        match self {
            MockValue::Integer(v) => Ok(v),
            other => Err(format!("Expected Integer, got {:?}", other)),
        }
    }

    /// Coercing version: converts Float/Boolean to Integer field element.
    fn to_int(&self) -> Result<BigInt, String> {
        match self {
            MockValue::Integer(v) => Ok(v.clone()),
            MockValue::Float(v) => Ok(to_field(&BigInt::from(*v as i64))),
            MockValue::Boolean(b) => Ok(bool_to_field(*b)),
            other => Err(format!("Cannot convert {:?} to Integer", other)),
        }
    }

    /// Get two values as integers, coercing if needed (for mixed-type IR ops).
    fn pair_as_ints(a: &MockValue, b: &MockValue) -> Result<(BigInt, BigInt), String> {
        Ok((a.to_int()?, b.to_int()?))
    }

    fn as_float(&self) -> Result<f64, String> {
        match self {
            MockValue::Float(v) => Ok(*v),
            MockValue::Integer(v) => Ok(from_signed(v).to_f64().unwrap_or(0.0)),
            other => Err(format!("Expected Float, got {:?}", other)),
        }
    }

    fn as_bool_or_int_truthy(&self) -> Result<bool, String> {
        match self {
            MockValue::Boolean(b) => Ok(*b),
            MockValue::Integer(v) => Ok(is_truthy(v)),
            other => Err(format!("Expected Boolean/Integer, got {:?}", other)),
        }
    }

    fn as_str(&self) -> Result<&str, String> {
        match self {
            MockValue::Str(s) => Ok(s),
            other => Err(format!("Expected Str, got {:?}", other)),
        }
    }
}

// ---------------------------------------------------------------------------
// Input entry (from Python)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct InputEntry {
    pub key: String,
    pub kind: String,
    pub value: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Mock execution result
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize)]
pub struct MockExecResult {
    pub satisfied: bool,
    pub assertion_failures: Vec<String>,
    pub public_outputs: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Mock executor
// ---------------------------------------------------------------------------

pub struct MockExecutor<'py> {
    values: Vec<MockValue>,
    memory: HashMap<u32, Vec<BigInt>>,
    inputs: HashMap<String, MockValue>,
    satisfied: bool,
    assertion_failures: Vec<String>,
    public_outputs: HashMap<String, serde_json::Value>,
    // For external function handling (preprocess phase)
    external_arg_store: HashMap<u32, Vec<MockValue>>,
    external_callables: Option<&'py Bound<'py, PyDict>>,
    py: Option<Python<'py>>,
}

impl<'py> MockExecutor<'py> {
    pub fn new(inputs: HashMap<String, MockValue>) -> Self {
        Self {
            values: Vec::new(),
            memory: HashMap::new(),
            inputs,
            satisfied: true,
            assertion_failures: Vec::new(),
            public_outputs: HashMap::new(),
            external_arg_store: HashMap::new(),
            external_callables: None,
            py: None,
        }
    }

    pub fn with_externals(
        mut self,
        py: Python<'py>,
        callables: &'py Bound<'py, PyDict>,
    ) -> Self {
        self.py = Some(py);
        self.external_callables = Some(callables);
        self
    }

    pub fn execute(&mut self, graph: &IRGraph) -> Result<MockExecResult, String> {
        self.values = vec![MockValue::None; graph.len()];

        for stmt in &graph.stmts {
            let args: Vec<MockValue> = stmt
                .arguments
                .iter()
                .map(|&id| self.values[id as usize].clone())
                .collect();

            let result = self.eval_ir(&stmt.ir, &args, stmt.stmt_id)?;
            self.values[stmt.stmt_id as usize] = result;
        }

        Ok(MockExecResult {
            satisfied: self.satisfied,
            assertion_failures: self.assertion_failures.clone(),
            public_outputs: self.public_outputs.clone(),
        })
    }

    fn eval_ir(
        &mut self,
        ir: &IR,
        args: &[MockValue],
        stmt_id: u32,
    ) -> Result<MockValue, String> {
        match ir {
            // ── Constants ──────────────────────────────────────────────
            IR::ConstantInt { value } => Ok(MockValue::Integer(to_field(&BigInt::from(*value)))),
            IR::ConstantFloat { value } => Ok(MockValue::Float(*value)),
            IR::ConstantBool { value } => Ok(MockValue::Boolean(*value)),
            IR::ConstantStr { value } => Ok(MockValue::Str(value.clone())),

            // ── Integer arithmetic (coerce Float args for IR gen compat) ─
            IR::AddI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(field_add(&a, &b)))
            }
            IR::SubI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(field_sub(&a, &b)))
            }
            IR::MulI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(field_mul(&a, &b)))
            }
            IR::DivI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(field_div(&a, &b)))
            }
            IR::FloorDivI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                let sa = from_signed(&a);
                let sb = from_signed(&b);
                if sb.is_zero() {
                    return Err("Division by zero".into());
                }
                Ok(MockValue::Integer(to_field(&sa.div_floor(&sb))))
            }
            IR::ModI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                let sa = from_signed(&a);
                let sb = from_signed(&b);
                if sb.is_zero() {
                    return Err("Modulo by zero".into());
                }
                Ok(MockValue::Integer(to_field(&sa.mod_floor(&sb))))
            }
            IR::PowI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(field_pow(&a, &b)))
            }
            IR::AbsI => {
                let a = args[0].to_int()?;
                let sa = from_signed(&a);
                Ok(MockValue::Integer(to_field(&sa.abs())))
            }
            IR::SignI => {
                let a = args[0].to_int()?;
                let sa = from_signed(&a);
                let sign = if sa > BigInt::zero() {
                    BigInt::one()
                } else if sa < BigInt::zero() {
                    to_field(&BigInt::from(-1))
                } else {
                    BigInt::zero()
                };
                Ok(MockValue::Integer(sign))
            }
            IR::InvI => {
                let a = args[0].to_int()?;
                Ok(MockValue::Integer(field_inv(&a)))
            }

            // ── Float arithmetic ───────────────────────────────────────
            IR::AddF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float(a + b))
            }
            IR::SubF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float(a - b))
            }
            IR::MulF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float(a * b))
            }
            IR::DivF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float(a / b))
            }
            IR::FloorDivF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float((a / b).floor()))
            }
            IR::ModF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float(a % b))
            }
            IR::PowF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Float(a.powf(b)))
            }
            IR::AbsF => {
                let a = args[0].as_float()?;
                Ok(MockValue::Float(a.abs()))
            }
            IR::SignF => {
                let a = args[0].as_float()?;
                Ok(MockValue::Float(if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    -1.0
                } else {
                    0.0
                }))
            }

            // ── Integer comparison (coerce Float args from buggy IR gen) ──
            IR::EqI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(bool_to_field(a == b)))
            }
            IR::NeI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(bool_to_field(a != b)))
            }
            IR::LtI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(bool_to_field(
                    from_signed(&a) < from_signed(&b),
                )))
            }
            IR::LteI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(bool_to_field(
                    from_signed(&a) <= from_signed(&b),
                )))
            }
            IR::GtI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(bool_to_field(
                    from_signed(&a) > from_signed(&b),
                )))
            }
            IR::GteI => {
                let (a, b) = MockValue::pair_as_ints(&args[0], &args[1])?;
                Ok(MockValue::Integer(bool_to_field(
                    from_signed(&a) >= from_signed(&b),
                )))
            }

            // ── Float comparison ───────────────────────────────────────
            IR::EqF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Integer(bool_to_field(a == b)))
            }
            IR::NeF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Integer(bool_to_field(a != b)))
            }
            IR::LtF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Integer(bool_to_field(a < b)))
            }
            IR::LteF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Integer(bool_to_field(a <= b)))
            }
            IR::GtF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Integer(bool_to_field(a > b)))
            }
            IR::GteF => {
                let (a, b) = (args[0].as_float()?, args[1].as_float()?);
                Ok(MockValue::Integer(bool_to_field(a >= b)))
            }

            // ── Math functions ─────────────────────────────────────────
            IR::SinF => Ok(MockValue::Float(args[0].as_float()?.sin())),
            IR::SinHF => Ok(MockValue::Float(args[0].as_float()?.sinh())),
            IR::CosF => Ok(MockValue::Float(args[0].as_float()?.cos())),
            IR::CosHF => Ok(MockValue::Float(args[0].as_float()?.cosh())),
            IR::TanF => Ok(MockValue::Float(args[0].as_float()?.tan())),
            IR::TanHF => Ok(MockValue::Float(args[0].as_float()?.tanh())),
            IR::SqrtF => Ok(MockValue::Float(args[0].as_float()?.sqrt())),
            IR::ExpF => Ok(MockValue::Float(args[0].as_float()?.exp())),
            IR::LogF => Ok(MockValue::Float(args[0].as_float()?.ln())),

            // ── Logical ────────────────────────────────────────────────
            IR::LogicalAnd => {
                let (a, b) = (
                    args[0].as_bool_or_int_truthy()?,
                    args[1].as_bool_or_int_truthy()?,
                );
                Ok(MockValue::Integer(bool_to_field(a && b)))
            }
            IR::LogicalOr => {
                let (a, b) = (
                    args[0].as_bool_or_int_truthy()?,
                    args[1].as_bool_or_int_truthy()?,
                );
                Ok(MockValue::Integer(bool_to_field(a || b)))
            }
            IR::LogicalNot => {
                let a = args[0].as_bool_or_int_truthy()?;
                Ok(MockValue::Integer(bool_to_field(!a)))
            }

            // ── Selection (mux) ────────────────────────────────────────
            IR::SelectI => {
                let cond = args[0].as_bool_or_int_truthy()?;
                Ok(if cond {
                    args[1].clone()
                } else {
                    args[2].clone()
                })
            }
            IR::SelectF => {
                let cond = args[0].as_bool_or_int_truthy()?;
                Ok(if cond {
                    args[1].clone()
                } else {
                    args[2].clone()
                })
            }
            IR::SelectB => {
                let cond = args[0].as_bool_or_int_truthy()?;
                Ok(if cond {
                    args[1].clone()
                } else {
                    args[2].clone()
                })
            }

            // ── Casting ────────────────────────────────────────────────
            IR::IntCast => {
                let v = args[0].as_float()?;
                Ok(MockValue::Integer(to_field(&BigInt::from(v as i64))))
            }
            IR::FloatCast => {
                let v = args[0].to_int()?;
                let sv = from_signed(&v);
                Ok(MockValue::Float(sv.to_f64().unwrap_or(0.0)))
            }
            IR::BoolCast => {
                let v = args[0].as_bool_or_int_truthy()?;
                Ok(MockValue::Integer(bool_to_field(v)))
            }

            // ── String operations ──────────────────────────────────────
            IR::AddStr => {
                let (a, b) = (args[0].as_str()?, args[1].as_str()?);
                Ok(MockValue::Str(format!("{}{}", a, b)))
            }
            IR::StrI => {
                let v = args[0].to_int()?;
                Ok(MockValue::Str(from_signed(&v).to_string()))
            }
            IR::StrF => {
                let v = args[0].as_float()?;
                Ok(MockValue::Str(v.to_string()))
            }

            // ── I/O ────────────────────────────────────────────────────
            IR::ReadInteger { indices, .. } => {
                let key = indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join("_");
                let val = self.inputs
                    .get(&key)
                    .ok_or_else(|| format!("Input not found for key: {}", key))?;
                // Coerce to integer if needed (e.g., Float param read as Integer)
                Ok(MockValue::Integer(val.to_int()?))
            }
            IR::ReadFloat { indices, .. } => {
                let key = indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join("_");
                let val = self.inputs
                    .get(&key)
                    .ok_or_else(|| format!("Input not found for key: {}", key))?;
                // Coerce to float if needed
                Ok(MockValue::Float(val.as_float()?))
            }
            IR::ReadHash { indices, .. } => {
                let key = indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join("_");
                self.inputs
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| format!("Hash input not found for key: {}", key))
            }
            IR::Print => {
                // args[0] = condition (Boolean/Integer), args[1] = string to print
                // Only print if condition is truthy, matching Python behavior.
                if args.len() >= 2 {
                    let cond = args[0].as_bool_or_int_truthy()?;
                    if cond {
                        if let Ok(s) = args[1].as_str() {
                            print!("{}", s);
                        }
                    }
                }
                Ok(MockValue::None)
            }

            // ── Assert ─────────────────────────────────────────────────
            IR::Assert => {
                let truthy = args[0].as_bool_or_int_truthy()?;
                if !truthy {
                    self.satisfied = false;
                    self.assertion_failures.push(format!(
                        "Assertion failed at stmt {}",
                        stmt_id
                    ));
                }
                Ok(MockValue::None)
            }

            // ── Expose public ──────────────────────────────────────────
            IR::ExposePublicI => {
                let v = args[0].to_int()?;
                let key = format!("public_i_{}", stmt_id);
                self.public_outputs.insert(
                    key,
                    serde_json::json!(from_signed(&v).to_string()),
                );
                Ok(MockValue::None)
            }
            IR::ExposePublicF => {
                let v = args[0].as_float()?;
                let key = format!("public_f_{}", stmt_id);
                self.public_outputs.insert(key, serde_json::json!(v));
                Ok(MockValue::None)
            }

            // ── Memory operations ──────────────────────────────────────
            IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            } => {
                let init = to_field(&BigInt::from(*init_value));
                self.memory
                    .insert(*segment_id, vec![init; *size as usize]);
                Ok(MockValue::None)
            }
            IR::WriteMemory { segment_id } => {
                let addr = args[0].to_int()?;
                let val = args[1].to_int()?;
                let idx = from_signed(&addr)
                    .to_usize()
                    .ok_or("Invalid memory address")?;
                if let Some(seg) = self.memory.get_mut(segment_id) {
                    if idx < seg.len() {
                        seg[idx] = val.clone();
                    } else {
                        return Err(format!(
                            "Memory write out of bounds: segment={}, addr={}",
                            segment_id, idx
                        ));
                    }
                } else {
                    return Err(format!("Memory segment {} not allocated", segment_id));
                }
                Ok(MockValue::None)
            }
            IR::ReadMemory { segment_id } => {
                let addr = args[0].to_int()?;
                let idx = from_signed(&addr)
                    .to_usize()
                    .ok_or("Invalid memory address")?;
                if let Some(seg) = self.memory.get(segment_id) {
                    if idx < seg.len() {
                        Ok(MockValue::Integer(seg[idx].clone()))
                    } else {
                        Err(format!(
                            "Memory read out of bounds: segment={}, addr={}",
                            segment_id, idx
                        ))
                    }
                } else {
                    Err(format!("Memory segment {} not allocated", segment_id))
                }
            }

            // ── Memory trace (no-op for mock) ──────────────────────────
            IR::MemoryTraceEmit { .. } => Ok(MockValue::None),
            IR::MemoryTraceSeal => Ok(MockValue::None),

            // ── Dynamic NDArray ────────────────────────────────────────
            IR::AllocateDynamicNDArrayMeta {
                array_id,
                max_length,
                ..
            } => {
                // Use memory segment keyed by array_id + offset
                let seg_id = 10000 + array_id;
                self.memory
                    .insert(seg_id, vec![BigInt::zero(); *max_length as usize]);
                Ok(MockValue::None)
            }
            IR::WitnessDynamicNDArrayMeta { .. } => Ok(MockValue::None),
            IR::AssertDynamicNDArrayMeta { .. } => Ok(MockValue::None),
            IR::DynamicNDArrayGetItem {
                array_id: _,
                segment_id,
            } => {
                let addr = args[0].to_int()?;
                let idx = from_signed(&addr).to_usize().ok_or("Invalid array index")?;
                let seg = self
                    .memory
                    .get(segment_id)
                    .ok_or(format!("Array segment {} not found", segment_id))?;
                if idx < seg.len() {
                    Ok(MockValue::Integer(seg[idx].clone()))
                } else {
                    Err(format!("Array index {} out of bounds", idx))
                }
            }
            IR::DynamicNDArraySetItem {
                array_id: _,
                segment_id,
            } => {
                let addr = args[0].to_int()?;
                let val = args[1].to_int()?;
                let idx = from_signed(&addr).to_usize().ok_or("Invalid array index")?;
                if let Some(seg) = self.memory.get_mut(segment_id) {
                    if idx < seg.len() {
                        seg[idx] = val.clone();
                    } else {
                        return Err(format!("Array index {} out of bounds", idx));
                    }
                } else {
                    return Err(format!("Array segment {} not found", segment_id));
                }
                Ok(MockValue::None)
            }

            // ── External function calls (preprocess phase) ─────────────
            IR::ExportExternalI {
                for_which,
                key: _,
                indices: _,
            } => {
                let val = args[0].clone();
                self.external_arg_store
                    .entry(*for_which)
                    .or_default()
                    .push(val);
                Ok(MockValue::None)
            }
            IR::ExportExternalF {
                for_which,
                key: _,
                indices: _,
            } => {
                let val = args[0].clone();
                self.external_arg_store
                    .entry(*for_which)
                    .or_default()
                    .push(val);
                Ok(MockValue::None)
            }
            IR::InvokeExternal {
                store_idx,
                func_name,
                ..
            } => {
                self.invoke_external(*store_idx, func_name)
            }

            // ── Hash (stub for mock) ───────────────────────────────────
            IR::PoseidonHash => {
                // Return a deterministic mock hash based on inputs
                let mut h = BigInt::from(0xcafebabe_u64);
                for arg in args {
                    if let Ok(v) = arg.to_int() {
                        h = field_add(&field_mul(&h, &BigInt::from(31)), &v);
                    }
                }
                Ok(MockValue::Integer(h))
            }
            IR::EqHash => {
                let (a, b) = (args[0].to_int()?, args[1].to_int()?);
                Ok(MockValue::Integer(bool_to_field(a == b)))
            }
        }
    }

    fn invoke_external(
        &mut self,
        store_idx: u32,
        func_name: &str,
    ) -> Result<MockValue, String> {
        let py = self
            .py
            .ok_or("Python interpreter not available for external calls")?;
        let callables = self
            .external_callables
            .ok_or("No external callables provided")?;

        let callable = callables
            .get_item(func_name)
            .map_err(|e| format!("Failed to get external '{}': {}", func_name, e))?
            .ok_or_else(|| format!("External function '{}' not found", func_name))?;

        // Collect accumulated args for this store
        let ext_args = self
            .external_arg_store
            .remove(&store_idx)
            .unwrap_or_default();

        // Convert MockValues to Python objects
        let py_args: Vec<PyObject> = ext_args
            .iter()
            .map(|v| match v {
                MockValue::Integer(i) => {
                    let sv = from_signed(i);
                    sv.to_i64()
                        .map(|n| n.into_pyobject(py).unwrap().unbind().into())
                        .unwrap_or_else(|| {
                            sv.to_string()
                                .into_pyobject(py)
                                .unwrap()
                                .unbind()
                                .into()
                        })
                }
                MockValue::Float(f) => f.into_pyobject(py).unwrap().unbind().into(),
                MockValue::Boolean(b) => {
                    let py_bool = (*b).into_pyobject(py).unwrap();
                    py_bool.to_owned().unbind().into()
                }
                MockValue::Str(s) => s.as_str().into_pyobject(py).unwrap().unbind().into(),
                MockValue::None => py.None(),
            })
            .collect();

        let py_tuple = PyTuple::new(py, &py_args)
            .map_err(|e| format!("Failed to create args tuple: {}", e))?;

        let result = callable
            .call1(&py_tuple)
            .map_err(|e| format!("External function '{}' failed: {}", func_name, e))?;

        // Convert result back to MockValue
        if let Ok(v) = result.extract::<i64>() {
            Ok(MockValue::Integer(to_field(&BigInt::from(v))))
        } else if let Ok(v) = result.extract::<f64>() {
            Ok(MockValue::Float(v))
        } else if let Ok(v) = result.extract::<bool>() {
            Ok(MockValue::Boolean(v))
        } else if let Ok(v) = result.extract::<String>() {
            Ok(MockValue::Str(v))
        } else {
            // Try to interpret as integer via Python int()
            if let Ok(v) = result.extract::<i128>() {
                Ok(MockValue::Integer(to_field(&BigInt::from(v))))
            } else {
                Err(format!(
                    "External function '{}' returned unsupported type",
                    func_name
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level mock_execute function (called from lib.rs PyO3)
// ---------------------------------------------------------------------------

pub fn run_mock_execute(
    py: Python<'_>,
    zk_program_ir_json: &str,
    preprocess_ir_json: &str,
    inputs_json: &str,
    external_callables: &Bound<'_, PyDict>,
) -> Result<String, String> {
    // 1. Parse inputs
    let entries: Vec<InputEntry> = serde_json::from_str(inputs_json)
        .map_err(|e| format!("Failed to parse inputs JSON: {}", e))?;

    let mut input_map: HashMap<String, MockValue> = HashMap::new();
    for entry in &entries {
        let val = match entry.kind.as_str() {
            "Integer" => {
                let v = entry
                    .value
                    .as_i64()
                    .map(|n| BigInt::from(n))
                    .or_else(|| {
                        entry
                            .value
                            .as_str()
                            .and_then(|s| BigInt::from_str(s).ok())
                    })
                    .ok_or_else(|| format!("Invalid integer value for key {}", entry.key))?;
                MockValue::Integer(to_field(&v))
            }
            "Float" => {
                let v = entry
                    .value
                    .as_f64()
                    .ok_or_else(|| format!("Invalid float value for key {}", entry.key))?;
                MockValue::Float(v)
            }
            "Hash" => {
                let v = entry
                    .value
                    .as_i64()
                    .map(BigInt::from)
                    .ok_or_else(|| format!("Invalid hash value for key {}", entry.key))?;
                MockValue::Integer(to_field(&v))
            }
            k => return Err(format!("Unknown input kind: {}", k)),
        };
        input_map.insert(entry.key.clone(), val);
    }

    // 2. Execute preprocess IR (may have external calls)
    let preprocess_data: Vec<serde_json::Value> = serde_json::from_str(preprocess_ir_json)
        .map_err(|e| format!("Failed to parse preprocess IR JSON: {}", e))?;
    let preprocess_graph = IRGraph::import_stmts(&preprocess_data)
        .map_err(|e| format!("Failed to import preprocess IR: {}", e))?;

    if !preprocess_graph.is_empty() {
        let mut preprocess_exec = MockExecutor::new(input_map.clone())
            .with_externals(py, external_callables);
        let _preprocess_result = preprocess_exec.execute(&preprocess_graph)?;

        // After preprocess, external results may have added new inputs.
        // The preprocess executor's values contain computed external results.
        // For now, we don't need to merge — the main IR's ReadInteger/ReadFloat
        // statements reference the same indices as circuit inputs. External
        // results flow through the preprocess IR and are already in the input map
        // if the preprocess produces them via ReadInteger/ReadFloat patterns.
    }

    // 3. Execute main zk_program IR
    let zk_data: Vec<serde_json::Value> = serde_json::from_str(zk_program_ir_json)
        .map_err(|e| format!("Failed to parse zk program IR JSON: {}", e))?;
    let zk_graph = IRGraph::import_stmts(&zk_data)
        .map_err(|e| format!("Failed to import zk program IR: {}", e))?;

    let mut main_exec = MockExecutor::new(input_map);
    let result = main_exec.execute(&zk_graph)?;

    serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IRStatement;

    #[test]
    fn test_bn254_field_add() {
        let a = BigInt::from(10);
        let b = BigInt::from(20);
        assert_eq!(field_add(&a, &b), BigInt::from(30));
    }

    #[test]
    fn test_bn254_field_sub_negative() {
        let a = BigInt::from(5);
        let b = BigInt::from(10);
        let result = field_sub(&a, &b);
        // 5 - 10 = -5 mod p = p - 5
        assert_eq!(result, bn254_prime() - BigInt::from(5));
        assert_eq!(from_signed(&result), BigInt::from(-5));
    }

    #[test]
    fn test_bn254_field_mul() {
        let a = BigInt::from(7);
        let b = BigInt::from(6);
        assert_eq!(field_mul(&a, &b), BigInt::from(42));
    }

    #[test]
    fn test_bn254_field_inv() {
        let a = BigInt::from(7);
        let inv_a = field_inv(&a);
        // a * inv(a) == 1
        assert_eq!(field_mul(&a, &inv_a), BigInt::one());
    }

    #[test]
    fn test_bn254_signed_interpretation() {
        let neg1 = to_field(&BigInt::from(-1));
        assert_eq!(neg1, bn254_prime() - BigInt::one());
        assert_eq!(from_signed(&neg1), BigInt::from(-1));
    }

    #[test]
    fn test_mock_execute_simple_add_assert() {
        // stmt0 = ReadInteger[0,0], stmt1 = ReadInteger[0,1]
        // stmt2 = AddI(0,1), stmt3 = Const(0), stmt4 = GtI(2,3), stmt5 = Assert(4)
        let stmts = vec![
            IRStatement::new(
                0,
                IR::ReadInteger {
                    indices: vec![0, 0],
                    is_public: true,
                },
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                IR::ReadInteger {
                    indices: vec![0, 1],
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
            IRStatement::new(3, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(4, IR::GtI, vec![2, 3], None),
            IRStatement::new(5, IR::Assert, vec![4], None),
        ];
        let graph = IRGraph::new(stmts);

        let mut inputs = HashMap::new();
        inputs.insert("0_0".to_string(), MockValue::Integer(BigInt::from(3)));
        inputs.insert("0_1".to_string(), MockValue::Integer(BigInt::from(4)));

        let mut exec = MockExecutor::new(inputs);
        let result = exec.execute(&graph).unwrap();
        assert!(result.satisfied);
    }

    #[test]
    fn test_mock_execute_assertion_failure() {
        // stmt0 = ReadInteger[0,0], stmt1 = Const(0), stmt2 = EqI(0,1), stmt3 = Assert(2)
        // Input is 5, asserting 5 == 0 should fail
        let stmts = vec![
            IRStatement::new(
                0,
                IR::ReadInteger {
                    indices: vec![0, 0],
                    is_public: true,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(2, IR::EqI, vec![0, 1], None),
            IRStatement::new(3, IR::Assert, vec![2], None),
        ];
        let graph = IRGraph::new(stmts);

        let mut inputs = HashMap::new();
        inputs.insert("0_0".to_string(), MockValue::Integer(BigInt::from(5)));

        let mut exec = MockExecutor::new(inputs);
        let result = exec.execute(&graph).unwrap();
        assert!(!result.satisfied);
    }

    #[test]
    fn test_mock_execute_negative_numbers() {
        // stmt0 = ReadInteger[0,0], stmt1 = ReadInteger[0,1]
        // stmt2 = AddI(0,1), stmt3 = Const(0), stmt4 = EqI(2,3), stmt5 = Assert(4)
        // Input: 1 + (-1) == 0
        let stmts = vec![
            IRStatement::new(
                0,
                IR::ReadInteger {
                    indices: vec![0, 0],
                    is_public: true,
                },
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                IR::ReadInteger {
                    indices: vec![0, 1],
                    is_public: true,
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
            IRStatement::new(3, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(4, IR::EqI, vec![2, 3], None),
            IRStatement::new(5, IR::Assert, vec![4], None),
        ];
        let graph = IRGraph::new(stmts);

        let mut inputs = HashMap::new();
        inputs.insert("0_0".to_string(), MockValue::Integer(BigInt::from(1)));
        inputs.insert(
            "0_1".to_string(),
            MockValue::Integer(to_field(&BigInt::from(-1))),
        );

        let mut exec = MockExecutor::new(inputs);
        let result = exec.execute(&graph).unwrap();
        assert!(result.satisfied);
    }
}
