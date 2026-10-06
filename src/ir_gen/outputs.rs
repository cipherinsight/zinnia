//! The circuit's outputs and the *output layout*: for every public value,
//! which output element (or shape entry) it is.
//!
//! **Outputs.** The outputs of the entry function are its return value. A
//! function that returns `None` (an *in-place kernel*) has as outputs the
//! array arguments its body writes (subscript or slice assignment, `op=`,
//! `out=`, or any other write into the argument's memory, also through views,
//! aliases and helpers), in parameter order, with their final contents; see
//! [`IRGenerator::in_place_outputs`].
//!
//! **Parts.** An output is split into *parts* the way the NumPy reference is
//! compared (`benchmarking/harness/outputs.py::reference_leaves`): the items
//! of a returned tuple, and the items of a list or tuple that holds arrays,
//! lists or tuples, are parts of their own (recursively); anything else
//! (an array, a scalar, a list of scalars) is one part. A part's elements are
//! its values in row-major order; a complex element is two cells, real then
//! imaginary.
//!
//! **Slots.** Each cell is either *exposed* (an `ExposePublic` statement; its
//! position in the public vector is resolved after optimization, see
//! [`OutputLayout::to_json`]) or a compile-time *constant* left out of the
//! public vector. A part with a runtime shape also records its runtime
//! length, rank and dimensions as slots, and how its payload cells are laid
//! out (compact row-major, or at static strides inside a larger buffer).
//!
//! This module decides which cells are exposed exactly as the circuit did
//! before the layout existed: the layout describes the public vector, it
//! does not change it.

use std::collections::BTreeSet;

use serde_json::{json, Value as Json};

use crate::ast::{ASTCircuit, ASTNode};
use crate::ir::ElidedPublic;
use crate::ir_defs::IR;
use crate::types::{DynamicNDArrayData, ScalarValue, Value};

use super::IRGenerator;

/// Number kind of an exposed public: integer (or boolean) or Q32 float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicKind {
    Int,
    Float,
}

impl PublicKind {
    fn code(self) -> &'static str {
        match self {
            PublicKind::Int => "i",
            PublicKind::Float => "f",
        }
    }
}

/// Where one cell of an output part comes from.
#[derive(Clone, Debug)]
pub enum Slot {
    /// The `ordinal`-th `ExposePublic` statement emitted by the circuit.
    Exposed { ordinal: usize, kind: PublicKind },
    /// A compile-time constant that is not in the public vector.
    Constant(ElidedPublic),
}

/// Which output a part belongs to.
#[derive(Clone, Debug)]
pub enum OutputSource {
    /// The return value; `path` indexes into nested returned tuples/lists
    /// (empty for a single returned value).
    Return { path: Vec<usize> },
    /// An array argument written by an in-place kernel.
    Argument { name: String },
}

/// Where the elements of a runtime-shaped part sit among its payload cells.
#[derive(Clone, Debug)]
pub enum CellLayout {
    /// Element `j` (row-major in the runtime shape) is cell `j`.
    Compact,
    /// Element with coordinates `c` is cell `offset + sum(c[k] * strides[k])`.
    Strided { offset: usize, strides: Vec<usize> },
}

/// The shape of one output part.
#[derive(Clone, Debug)]
pub enum PartShape {
    Static {
        shape: Vec<usize>,
    },
    /// A runtime-shaped array: the first `rank` of `dims` are its shape and
    /// their product is `length`. Shape entries computed by the same
    /// statement share one slot (`ExposedMetadata`): a rank-1 array's dim is
    /// its length's public value.
    Dynamic {
        length: Slot,
        rank: Slot,
        dims: Vec<Slot>,
        layout: CellLayout,
    },
}

/// One output part: its source, shape and cells.
#[derive(Clone, Debug)]
pub struct OutputPart {
    pub source: OutputSource,
    pub shape: PartShape,
    /// Complex elements: two cells (real, imaginary) per element.
    pub complex: bool,
    pub cells: Vec<Slot>,
}

/// The layout of every output part, in output order, plus the kind of every
/// `ExposePublic` statement in emission order.
#[derive(Clone, Debug, Default)]
pub struct OutputLayout {
    pub parts: Vec<OutputPart>,
    pub exposure_kinds: Vec<PublicKind>,
}

impl OutputLayout {
    /// Render as JSON, given the public-vector index of every exposure
    /// (`public_index[ordinal]`, from the optimized IR). Format:
    ///
    /// ```text
    /// slot    = {"public": i, "kind": "i" | "f"} | {"constant": v}
    /// output  = {"return": [path...]} | {"argument": name}
    /// static  = {"output": output, "shape": [d...], "cells": [slot...]}
    /// dynamic = {"output": output, "dynamic": {"length": slot, "rank": slot,
    ///            "dims": [slot...], "cells": [slot...],
    ///            ("strides": [s...], "offset": o)?}}
    /// ```
    ///
    /// plus `"complex": true` on a part whose elements are complex.
    pub fn to_json(&self, public_index: &[usize]) -> Json {
        let slot = |s: &Slot| -> Json {
            match s {
                Slot::Exposed { ordinal, kind } => {
                    json!({"public": public_index[*ordinal], "kind": kind.code()})
                }
                Slot::Constant(ElidedPublic::Int(v)) => json!({"constant": v}),
                Slot::Constant(ElidedPublic::Float(v)) => json!({"constant": v}),
            }
        };
        let slots = |cells: &[Slot]| -> Json { Json::Array(cells.iter().map(slot).collect()) };
        let parts = self
            .parts
            .iter()
            .map(|part| {
                let output = match &part.source {
                    OutputSource::Return { path } => json!({"return": path}),
                    OutputSource::Argument { name } => json!({"argument": name}),
                };
                let mut out = json!({"output": output});
                match &part.shape {
                    PartShape::Static { shape } => {
                        out["shape"] = json!(shape);
                        out["cells"] = slots(&part.cells);
                    }
                    PartShape::Dynamic {
                        length,
                        rank,
                        dims,
                        layout,
                    } => {
                        let mut dynamic = json!({
                            "length": slot(length),
                            "rank": slot(rank),
                            "dims": slots(dims),
                            "cells": slots(&part.cells),
                        });
                        if let CellLayout::Strided { offset, strides } = layout {
                            dynamic["strides"] = json!(strides);
                            dynamic["offset"] = json!(offset);
                        }
                        out["dynamic"] = dynamic;
                    }
                }
                if part.complex {
                    out["complex"] = json!(true);
                }
                out
            })
            .collect();
        Json::Array(parts)
    }
}

/// Compute the public-vector position of every exposure in an optimized
/// graph and describe its public inputs. Returns `(public_index, kinds,
/// public_inputs)`: `public_index[k]` is the position of the `k`-th
/// `ExposePublic` statement, `kinds[k]` its kind; `public_inputs` lists the
/// inputs read as public values as `{"public": i, "input": path, "kind"}`.
/// The public vector is every public input read and every exposure, in
/// statement order (the halo2-base instance column; the mock reports the
/// exposures only).
pub fn public_vector_positions(stmts: &[crate::ir::IRStatement]) -> (Vec<usize>, Vec<PublicKind>, Vec<Json>) {
    let mut public_index = Vec::new();
    let mut kinds = Vec::new();
    let mut inputs = Vec::new();
    let mut position = 0usize;
    for stmt in stmts {
        match &stmt.ir {
            IR::ReadInteger { path, is_public: true } | IR::ReadHash { path, is_public: true } => {
                inputs.push(json!({"public": position, "input": path.display(), "kind": "i"}));
                position += 1;
            }
            IR::ReadFloat { path, is_public: true } => {
                inputs.push(json!({"public": position, "input": path.display(), "kind": "f"}));
                position += 1;
            }
            IR::ExposePublicI | IR::ExposePublicF => {
                public_index.push(position);
                kinds.push(if matches!(stmt.ir, IR::ExposePublicF) {
                    PublicKind::Float
                } else {
                    PublicKind::Int
                });
                position += 1;
            }
            _ => {}
        }
    }
    (public_index, kinds, inputs)
}

/// Whether `value` is an array, list or tuple (a container in the sense of
/// the part-splitting rule).
fn is_container(value: &Value) -> bool {
    matches!(
        value,
        Value::StaticArray { .. }
            | Value::DynamicNDArray(_)
            | Value::NDArray(_)
            | Value::List(_)
            | Value::Tuple(_)
    )
}

/// Split a returned value into parts (see the module docs).
fn return_parts(value: &Value) -> Vec<(Vec<usize>, Value)> {
    match value {
        Value::Tuple(data) => data
            .values
            .iter()
            .enumerate()
            .flat_map(|(i, item)| part_or_split(item, vec![i]))
            .collect(),
        other => part_or_split(other, Vec::new()),
    }
}

fn part_or_split(value: &Value, path: Vec<usize>) -> Vec<(Vec<usize>, Value)> {
    match value {
        Value::List(data) | Value::Tuple(data) if data.values.iter().any(is_container) => data
            .values
            .iter()
            .enumerate()
            .flat_map(|(i, item)| {
                let mut sub = path.clone();
                sub.push(i);
                part_or_split(item, sub)
            })
            .collect(),
        other => vec![(path, other.clone())],
    }
}

/// Whether any `return` in `block` (outside nested chips, which are separate
/// ASTs) returns a value other than `None`.
fn block_returns_value(block: &[ASTNode]) -> bool {
    block.iter().any(|stmt| match stmt {
        ASTNode::ASTReturnStatement(ret) => {
            !matches!(ret.expr.as_deref(), None | Some(ASTNode::ASTConstantNone(_)))
        }
        ASTNode::ASTCondStatement(cond) => {
            block_returns_value(&cond.t_block) || block_returns_value(&cond.f_block)
        }
        ASTNode::ASTForInStatement(stmt) => {
            block_returns_value(&stmt.block) || block_returns_value(&stmt.orelse)
        }
        ASTNode::ASTWhileStatement(stmt) => {
            block_returns_value(&stmt.block) || block_returns_value(&stmt.orelse)
        }
        _ => false,
    })
}

/// An array argument of an in-place kernel, as bound at circuit entry.
#[derive(Clone, Debug)]
pub(crate) struct ArgumentArray {
    pub name: String,
    pub initial: Value,
}

/// The memory segments holding an array value's cells.
fn array_segments(value: &Value) -> Vec<u32> {
    match value {
        Value::StaticArray {
            segment_id,
            imag_segment_id,
            ..
        } => std::iter::once(*segment_id).chain(*imag_segment_id).collect(),
        Value::DynamicNDArray(data) => vec![data.segment_id],
        _ => Vec::new(),
    }
}

impl IRGenerator {
    /// Whether the entry function is an in-place kernel (returns no value).
    /// Its array arguments are visible to the caller, so `a op= v` on one
    /// writes into it, as in Python.
    pub(crate) fn begin_in_place_kernel(&mut self, n: &ASTCircuit) {
        if block_returns_value(&n.block) {
            return;
        }
        for input in &n.inputs {
            if let Some(value) = self.ctx.get(&input.name) {
                if matches!(value, Value::StaticArray { .. } | Value::DynamicNDArray(_)) {
                    self.in_place_arguments.push(ArgumentArray {
                        name: input.name.clone(),
                        initial: value,
                    });
                }
            }
        }
        self.in_place_body_start = self.builder.stmts.len();
    }

    /// Whether `segment` holds an array argument of an in-place kernel.
    pub(crate) fn is_caller_visible_segment(&self, segment: u32) -> bool {
        self.in_place_arguments
            .iter()
            .any(|arg| array_segments(&arg.initial).contains(&segment))
    }

    /// The outputs of an in-place kernel: every array argument the body
    /// wrote, in parameter order, with its final contents.
    ///
    /// The caller's array lives in the argument's memory segment for the
    /// whole run: every in-place update of it (subscript and slice
    /// assignment, `op=` via [`Self::update_in_place`], `out=`, writes through
    /// views, aliases and helpers) writes into that segment, under the write
    /// predicate of the statement that makes it (see
    /// [`Self::write_back_if_shared`] for updates that produce a new array
    /// value). Rebinding the parameter name (`A = A + 1`, or swapping two
    /// names) makes the name denote another object and leaves the caller's
    /// array alone, as in Python. So an argument is written exactly when a
    /// memory write into its segment was emitted after circuit entry, and its
    /// output is the segment's final contents read with the argument's
    /// original shape. An argument whose segment was marked stale (a write
    /// through a copy that NumPy would treat as a view) is rejected.
    pub(crate) fn in_place_outputs(&mut self) -> Vec<(OutputSource, Value)> {
        let written_segments: BTreeSet<u32> = self.builder.stmts[self.in_place_body_start..]
            .iter()
            .filter_map(|stmt| match stmt.ir {
                IR::WriteMemory { segment_id } => Some(segment_id),
                _ => None,
            })
            .collect();
        let mut outputs = Vec::new();
        for arg in std::mem::take(&mut self.in_place_arguments) {
            self.check_not_stale(&arg.name, &arg.initial);
            let written = array_segments(&arg.initial)
                .iter()
                .any(|s| written_segments.contains(s));
            if written {
                outputs.push((OutputSource::Argument { name: arg.name }, arg.initial));
            }
        }
        outputs
    }

    /// After an in-place update of the array `current` (bound to `name`)
    /// produced `updated` in another memory segment: when another binding or
    /// the caller (an argument of an in-place kernel) still holds `current`,
    /// write `updated` back into it and return `current`, so every holder
    /// sees the update, as Python's single object would. Returns `None` when
    /// rebinding `name` to `updated` is equivalent.
    pub(crate) fn write_back_if_shared(&mut self, name: &str, current: &Value, updated: &Value) -> Option<Value> {
        let segment = super::detached_copies::array_segment(current)?;
        let shared = self.is_caller_visible_segment(segment)
            || self
                .ctx
                .find_other_binding(name, |v| super::detached_copies::array_segment(v) == Some(segment))
                .is_some();
        if !shared {
            return None;
        }
        if !matches!(current, Value::StaticArray { .. }) {
            panic!(
                "this in-place update of the bounded-dynamic array `{}` is not supported while \
                 another name (or the caller) refers to the same array",
                name
            );
        }
        Some(super::named_attr::write_into(&mut self.builder, updated, current))
    }

    /// Expose the return value: every part in order.
    pub(crate) fn expose_return_value(&mut self, value: &Value) {
        for (path, part) in return_parts(value) {
            self.expose_part(OutputSource::Return { path }, &part);
        }
    }

    /// Expose one output part and record its layout.
    pub(crate) fn expose_part(&mut self, source: OutputSource, value: &Value) {
        let part = match value {
            Value::StaticArray { segment_id, .. }
                if self
                    .builder
                    .provenance
                    .runtime_extents_of(*segment_id)
                    .is_some() =>
            {
                match self.expose_runtime_extents_array(source.clone(), value) {
                    Some(part) => part,
                    None => {
                        let promoted =
                            crate::helpers::promote::promote_static_to_dynamic_with_extents(
                                &mut self.builder,
                                value,
                            );
                        return self.expose_part(source, &promoted);
                    }
                }
            }
            Value::DynamicNDArray(data) => self.expose_dynamic_array(source, data),
            Value::StaticArray { shape, .. } => {
                let shape = shape.clone();
                let leaves =
                    crate::helpers::composite::flatten_composite_with_builder(&mut self.builder, value);
                self.static_part(source, shape, &leaves)
            }
            Value::List(data) | Value::Tuple(data) => {
                // A list or tuple of scalars (containers were split off).
                let shape = vec![data.values.len()];
                let leaves = data.values.clone();
                self.static_part(source, shape, &leaves)
            }
            Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. } => {
                self.static_part(source, Vec::new(), std::slice::from_ref(value))
            }
            Value::None => return,
            other => panic!(
                "a circuit output of type {} is not supported (outputs are numbers, arrays, and \
                 lists or tuples of them)",
                other.zinnia_type()
            ),
        };
        self.output_layout.parts.push(part);
    }

    fn static_part(&mut self, source: OutputSource, shape: Vec<usize>, leaves: &[Value]) -> OutputPart {
        let complex = leaves.iter().any(|leaf| matches!(leaf, Value::Complex { .. }));
        let mut cells = Vec::new();
        for leaf in leaves {
            cells.extend(self.expose_cell(leaf, complex));
        }
        OutputPart {
            source,
            shape: PartShape::Static { shape },
            complex,
            cells,
        }
    }

    /// Expose one element. A compile-time constant number is left out of the
    /// public vector (it is fixed by the program text) and recorded as a
    /// constant; complex parts are always exposed. In a complex part a real
    /// element gets a constant zero imaginary cell, as NumPy's conversion
    /// to a complex array does.
    fn expose_cell(&mut self, leaf: &Value, complex_part: bool) -> Vec<Slot> {
        let constant = |builder: &mut crate::builder::IRBuilder, c: ElidedPublic| {
            builder.elided_publics.push(c.clone());
            Slot::Constant(c)
        };
        let mut slots = match leaf {
            Value::Integer(sv) if sv.static_val.is_some() => {
                vec![constant(&mut self.builder, ElidedPublic::Int(sv.static_val.unwrap()))]
            }
            Value::Boolean(sv) if sv.static_val.is_some() => vec![constant(
                &mut self.builder,
                ElidedPublic::Int(i64::from(sv.static_val.unwrap())),
            )],
            Value::Float(sv) if sv.static_val.is_some() => {
                vec![constant(&mut self.builder, ElidedPublic::Float(sv.static_val.unwrap()))]
            }
            Value::Integer(_) | Value::Boolean(_) => vec![self.expose(leaf, PublicKind::Int)],
            Value::Float(_) => vec![self.expose(leaf, PublicKind::Float)],
            Value::Complex { real, imag } => {
                return vec![
                    self.expose(&Value::Float(real.clone()), PublicKind::Float),
                    self.expose(&Value::Float(imag.clone()), PublicKind::Float),
                ];
            }
            other => panic!(
                "a circuit output element of type {} is not supported",
                other.zinnia_type()
            ),
        };
        if complex_part {
            slots.push(Slot::Constant(ElidedPublic::Float(0.0)));
        }
        slots
    }

    fn expose(&mut self, value: &Value, kind: PublicKind) -> Slot {
        match kind {
            PublicKind::Int => self.builder.ir_expose_public_i(value),
            PublicKind::Float => self.builder.ir_expose_public_f(value),
        };
        let ordinal = self.output_layout.exposure_kinds.len();
        self.output_layout.exposure_kinds.push(kind);
        Slot::Exposed { ordinal, kind }
    }

    /// A shape entry: a constant when known at compile time and `elide`
    /// allows it, otherwise exposed. Metadata constants are not recorded as
    /// elided outputs (they are not output elements).
    fn metadata_slot(&mut self, value: &ScalarValue<i64>, elide: bool) -> Slot {
        match value.static_val {
            Some(v) if elide => Slot::Constant(ElidedPublic::Int(v)),
            _ => self.expose(&Value::Integer(value.clone()), PublicKind::Int),
        }
    }

    /// A static array that carries runtime extents (a slice with runtime
    /// bounds): its runtime shape is the extents, its cells the envelope
    /// rectangle `shape`. Returns `None` when the extents are not available
    /// as values (the caller then exposes it as a dynamic array).
    fn expose_runtime_extents_array(&mut self, source: OutputSource, value: &Value) -> Option<OutputPart> {
        let Value::StaticArray {
            segment_id, shape, ..
        } = value
        else {
            return None;
        };
        let extents = self.builder.provenance.runtime_extents_of(*segment_id)?.clone();
        let extent_values: Option<Vec<Value>> =
            extents.iter().map(|vid| self.builder.value_of_vid(*vid)).collect();
        let extent_values = extent_values.filter(|values| values.len() == shape.len())?;
        let shape = shape.clone();
        let total: usize = shape.iter().product();
        let length = extent_values
            .iter()
            .cloned()
            .reduce(|acc, ext| self.builder.ir_mul_i(&acc, &ext))
            .unwrap_or_else(|| self.builder.ir_constant_int(0));
        // The length is left out when it is the constant envelope size; a
        // constant extent is left out.
        let length_slot = match length.int_val() {
            Some(v) if v == total as i64 => Slot::Constant(ElidedPublic::Int(v)),
            _ => self.expose(&length, PublicKind::Int),
        };
        let mut exposed = ExposedMetadata::default();
        exposed.note(&length, &length_slot);
        let dims = extent_values
            .iter()
            .map(|ext| match ext.int_val() {
                Some(v) => Slot::Constant(ElidedPublic::Int(v)),
                None => match exposed.slot_of(ext) {
                    Some(slot) => slot,
                    None => {
                        let slot = self.expose(ext, PublicKind::Int);
                        exposed.note(ext, &slot);
                        slot
                    }
                },
            })
            .collect();
        let leaves = crate::helpers::composite::flatten_composite_with_builder(&mut self.builder, value);
        let complex = leaves.iter().any(|leaf| matches!(leaf, Value::Complex { .. }));
        let mut cells = Vec::new();
        for leaf in &leaves {
            cells.extend(self.expose_cell(leaf, complex));
        }
        Some(OutputPart {
            source,
            shape: PartShape::Dynamic {
                length: length_slot,
                rank: Slot::Constant(ElidedPublic::Int(shape.len() as i64)),
                dims,
                layout: CellLayout::Strided {
                    offset: 0,
                    strides: row_major_strides(&shape),
                },
            },
            complex,
            cells,
        })
    }

    /// A runtime-shaped array: runtime length, rank and dimensions (each left
    /// out when it is a compile-time constant; the length only when it is the
    /// buffer size), then every payload cell of the buffer.
    fn expose_dynamic_array(&mut self, source: OutputSource, data: &DynamicNDArrayData) -> OutputPart {
        let max_length = data.max_length();
        let length_is_capacity = data.meta.runtime_length.static_val == Some(max_length as i64);
        let length = self.metadata_slot(&data.meta.runtime_length, length_is_capacity);
        let rank = self.metadata_slot(&data.meta.runtime_rank, true);
        let mut exposed = ExposedMetadata::default();
        exposed.note(&Value::Integer(data.meta.runtime_length.clone()), &length);
        let dims = (0..data.max_rank())
            .map(|axis| {
                let dim = data
                    .meta
                    .runtime_shape
                    .get(axis)
                    .cloned()
                    .unwrap_or_else(|| ScalarValue::constant(0));
                let dim_value = Value::Integer(dim.clone());
                if let Some(slot) = exposed.slot_of(&dim_value) {
                    return slot;
                }
                let slot = self.metadata_slot(&dim, true);
                exposed.note(&dim_value, &slot);
                slot
            })
            .collect();
        let mut cells = Vec::with_capacity(max_length);
        let complex = matches!(data.dtype, crate::types::NumberType::Complex);
        for i in 0..max_length {
            let addr = self.builder.ir_constant_int(i as i64);
            let cell =
                crate::helpers::segment::read_typed(&mut self.builder, data.segment_id, &addr, data.dtype);
            cells.extend(self.expose_cell(&cell, complex));
        }
        OutputPart {
            source,
            shape: PartShape::Dynamic {
                length,
                rank,
                dims,
                layout: dynamic_cell_layout(data),
            },
            complex,
            cells,
        }
    }
}

/// The shape entries of one runtime-shaped part already in the public
/// vector, by the IR statement that computes them. A rank-1 array's only
/// dimension is its length, the same statement: exposing it twice would put
/// the same value in two public slots (two public inputs to pay for in the
/// circuit, and nothing the verifier learns from the second). Its slot
/// repeats the first one instead; a decoder reads the same public value for
/// both entries, so the decoded shape is unchanged.
#[derive(Default)]
struct ExposedMetadata {
    slots: Vec<(crate::types::StmtId, Slot)>,
}

impl ExposedMetadata {
    /// Record `slot` as the public slot of `value` (a runtime value only).
    fn note(&mut self, value: &Value, slot: &Slot) {
        if let (Some(stmt), Slot::Exposed { .. }) = (value.stmt_id(), slot) {
            self.slots.push((stmt, slot.clone()));
        }
    }

    /// The slot already exposing the statement that computes `value`.
    fn slot_of(&self, value: &Value) -> Option<Slot> {
        let stmt = value.stmt_id()?;
        self.slots.iter().find(|(s, _)| *s == stmt).map(|(_, slot)| slot.clone())
    }
}

/// Row-major strides of `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// How a runtime-shaped array's elements sit in its buffer. Its element
/// addresses are `offset + sum(c[k] * stride[k])` over the runtime strides;
/// when those are compile-time constants they are the layout. Otherwise the
/// strides are computed from the runtime shape, which is the compact
/// row-major layout.
fn dynamic_cell_layout(data: &DynamicNDArrayData) -> CellLayout {
    let offset = data.meta.runtime_offset.static_val;
    let strides: Option<Vec<usize>> = data
        .meta
        .runtime_strides
        .iter()
        .map(|s| s.static_val.and_then(|v| usize::try_from(v).ok()))
        .collect();
    match (offset, strides) {
        (Some(offset), Some(strides)) if strides.len() == data.max_rank() => CellLayout::Strided {
            offset: offset.max(0) as usize,
            strides,
        },
        _ => CellLayout::Compact,
    }
}
