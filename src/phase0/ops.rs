//! The operator table: name to shape class (`coverage.md` §5).
//!
//! `coverage.md` §5 groups the baseline's whole operator surface into shape
//! classes, one rule family each. This module is the table that assigns a
//! class to every dispatched name; the class handlers live in
//! `crate::phase0::walk` and call the transfer rules of
//! [`crate::phase0::rules`]. Adding a name the baseline already accepts is
//! one row here.
//!
//! The names are copied from the baseline's dispatch tables
//! (`src/ir_gen/named_attr/*.rs`, `src/ops/registry.rs`,
//! `src/helpers/value_ops.rs`), which are data. Nothing in `src/ir_gen` is
//! imported: phase 0 is independent of the IR generator
//! (`implementation.md` §1).
//!
//! A name the baseline dispatches but that no program of the parity set uses
//! may be absent; the walker then answers
//! [`crate::phase0::Reject::UnsupportedOperator`], which is what the coverage
//! histogram reports.

use crate::phase0::desc::Dtype;
use crate::phase0::rules::ReduceKind;

/// The result dtype of an elementwise unary operator (`E-Map`, §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOut {
    /// The operand's dtype, unchanged (`negative`, `abs`, `floor`).
    Same,
    /// A real result whatever the operand (`sqrt`, `exp`, `sin`).
    Float,
    /// A predicate (`logical_not`).
    Bool,
    /// The real part of a complex operand (`real`, `imag`, `abs` on complex).
    RealPart,
}

/// The domain of an elementwise unary operator: `E-Map`'s `defcond_f`
/// (`definition.md` §5.3, "`defcond_f` is the domain condition of `f`").
///
/// Only the operators that *raise* outside their domain carry one. The
/// `math.*` forms do — `math.sqrt(-1)` is a `ValueError` — and the `np.*`
/// ufuncs return `nan` with a warning instead, but a condition that is true
/// on every successful run costs nothing and is what the circuit has to
/// enforce either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Every value is in the domain: `exp`, `sin`, `floor`, and all wrapping
    /// integer arithmetic.
    All,
    /// `x ≥ 0`: `sqrt`.
    NonNegative,
    /// `x > 0`: `log`.
    Positive,
    /// `-1 ≤ x ≤ 1`: `arccos`, `arcsin`.
    UnitInterval,
}

/// An elementwise binary operator (`E-Zip`, §5.3, and `E-Arith` / `E-Cmp`,
/// §5.2, when both operands are scalars).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZipOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Min,
    Max,
    Atan2,
}

impl ZipOp {
    /// The operator name in `ASTBinaryOperator.operator`
    /// (`zinnia/compile/transformer/base.py`'s `get_op_name_from_node`).
    pub fn from_ast(operator: &str) -> Option<ZipOp> {
        Some(match operator {
            "add" => ZipOp::Add,
            "sub" => ZipOp::Sub,
            "mul" => ZipOp::Mul,
            "div" => ZipOp::Div,
            "floor_div" => ZipOp::FloorDiv,
            "mod" => ZipOp::Mod,
            "pow" => ZipOp::Pow,
            "eq" => ZipOp::Eq,
            "ne" => ZipOp::Ne,
            "lt" => ZipOp::Lt,
            "lte" => ZipOp::Le,
            "gt" => ZipOp::Gt,
            "gte" => ZipOp::Ge,
            "and" => ZipOp::And,
            "or" => ZipOp::Or,
            "bit_and" => ZipOp::BitAnd,
            "bit_or" => ZipOp::BitOr,
            "bit_xor" => ZipOp::BitXor,
            "shl" => ZipOp::Shl,
            "shr" => ZipOp::Shr,
            _ => return None,
        })
    }

    /// Whether the operator yields a Boolean rather than a number.
    pub fn is_predicate(self) -> bool {
        matches!(
            self,
            ZipOp::Eq
                | ZipOp::Ne
                | ZipOp::Lt
                | ZipOp::Le
                | ZipOp::Gt
                | ZipOp::Ge
                | ZipOp::And
                | ZipOp::Or
                | ZipOp::Xor
        )
    }

    /// The scalar arithmetic rule for this operator (§5.2's `E-Arith`), when
    /// it has one.
    pub fn arith(self) -> Option<crate::phase0::rules::ArithOp> {
        use crate::phase0::rules::ArithOp;
        Some(match self {
            ZipOp::Add => ArithOp::Add,
            ZipOp::Sub => ArithOp::Sub,
            ZipOp::Mul => ArithOp::Mul,
            ZipOp::Div => ArithOp::Div,
            ZipOp::FloorDiv => ArithOp::FloorDiv,
            ZipOp::Mod => ArithOp::Mod,
            ZipOp::Pow => ArithOp::Pow,
            _ => return None,
        })
    }

    /// The scalar comparison rule for this operator (§5.2's `E-Cmp`), when it
    /// has one.
    pub fn cmp(self) -> Option<crate::phase0::rules::CmpOp> {
        use crate::phase0::rules::CmpOp;
        Some(match self {
            ZipOp::Eq => CmpOp::Eq,
            ZipOp::Ne => CmpOp::Ne,
            ZipOp::Lt => CmpOp::Lt,
            ZipOp::Le => CmpOp::Le,
            ZipOp::Gt => CmpOp::Gt,
            ZipOp::Ge => CmpOp::Ge,
            _ => return None,
        })
    }

    /// The result dtype of `a op b`.
    ///
    /// A predicate yields `Bool`; Python's `/` yields a real (`coverage.md`
    /// §9: the baseline floor-divides integers, which phase 0 does not
    /// replicate — it has no shape effect either way); everything else
    /// promotes.
    pub fn out_dtype(self, a: Dtype, b: Dtype) -> Option<Dtype> {
        if self.is_predicate() {
            return Some(Dtype::Bool);
        }
        let promoted = Dtype::promote(a, b)?;
        Some(match self {
            ZipOp::Div => match promoted {
                Dtype::Complex => Dtype::Complex,
                _ => Dtype::Float,
            },
            _ => promoted,
        })
    }
}

/// Which member of the `vstack` / `hstack` / `dstack` / `column_stack`
/// family a name is (`coverage.md` §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackKind {
    /// `vstack`, `row_stack`: promote to rank 2, concatenate on axis 0.
    Vertical,
    /// `hstack`: axis 1 when any operand has rank >= 2, else axis 0.
    Horizontal,
    /// `dstack`: promote to rank 3, concatenate on axis 2.
    Depth,
    /// `column_stack`: rank-1 operands become columns.
    Column,
}

/// The properties and no-argument accessors of `E-Property` (§5.11) plus the
/// complex accessors the baseline puts next to them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prop {
    Shape,
    Ndim,
    Size,
    Dtype,
    /// `.T`: the full axis reversal of `E-Transpose`.
    Transposed,
    /// `.flat`: the flattening of `E-Reshape`.
    Flat,
    /// `.real`, `.conjugate`: identity on the shape.
    RealPart,
    /// `.imag`: identity on the shape, real dtype.
    ImagPart,
}

/// The list methods of `S-ListMethod` (§5.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListMethod {
    Append,
    Extend,
    Insert,
    Pop,
    Remove,
    Clear,
    Reverse,
    Copy,
    Count,
    Index,
}

impl ListMethod {
    /// Whether the method changes the length of its receiver.
    ///
    /// §3.3 makes a composite's length a compile-time constant, so a
    /// length-changing method is only meaningful where the path condition
    /// folds to `true`; the walker rejects the rest
    /// ([`crate::phase0::walk`]).
    pub fn changes_length(self) -> bool {
        matches!(
            self,
            ListMethod::Append
                | ListMethod::Extend
                | ListMethod::Insert
                | ListMethod::Pop
                | ListMethod::Remove
                | ListMethod::Clear
        )
    }
}

/// The shape class of one operator name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    // §5.1, §5.3: elementwise
    Map(MapOut, Domain),
    Zip(ZipOp),
    /// `np.<binop>_outer`: both operands flattened, result rank 2.
    Outer,
    Clip,
    IsClose,
    /// `np.allclose`, `np.array_equal`: one Boolean.
    Predicate,

    // §5.2, §5.7: reductions
    Reduce(ReduceKind),
    /// `cumsum`, `cumprod`: dims unchanged.
    Cumulative,
    Percentile,

    // §5.6, §5.7: shape transforms
    Reshape,
    /// `flatten`, `ravel`, `.flat`: rank 1 of the same size.
    Ravel,
    /// `tolist`: the nested list Python returns (`coverage.md` §5.3).
    ToList,
    Transpose,
    Swapaxes,
    Moveaxis,
    Squeeze,
    ExpandDims,
    BroadcastTo,
    AtLeast(usize),
    /// `flip`, `flipud`, `fliplr`: dims unchanged (`E-Reverse`).
    Reverse,
    Rot90,
    Concat,
    Stack,
    StackFamily(StackKind),
    Diag,
    Delete,
    Insert,
    Pad,
    Repeat,
    Tile,
    Diff,
    Sort {
        index_result: bool,
    },
    Unique,

    // §5.8: constructors
    /// `zeros`, `ones`, `empty`, `ndarray`: shape argument, `dtype` kwarg.
    Zeros,
    Full,
    /// `zeros_like` and friends: the prototype's shape.
    Like,
    Identity,
    Arange,
    Linspace,
    /// `np.array`, `np.asarray`: a composite becomes an array.
    Asarray,
    Indices,

    // §5.4, §5.5: indexing and gather
    Take,
    TakeAlongAxis,
    Where,
    Nonzero,
    Argwhere,
    Flatnonzero,
    In1d,
    Bincount,
    UnravelIndex,
    RavelMultiIndex,
    Filter,

    // §5.7: linear algebra and statistics
    Matmul,
    Tensordot,
    Einsum,
    Corrcoef,
    Correlate,
    LinalgNorm,

    // §5.11 and the builtins of `coverage.md` §5.7, §5.8
    /// `int(x)`, `float(x)`, `bool(x)`, `np.<dtype>(x)` name their target
    /// dtype; `astype` carries `None`, meaning "take it from argument 1".
    Cast(Option<Dtype>),
    /// A dtype object used as a value (`np.float64`), the baseline's
    /// `Value::Class`.
    DtypeClass(Dtype),
    /// `np.pi`, `np.e`, `math.pi`: a real constant, no term (§3.2).
    FloatConstant,
    Len,
    Range,
    Enumerate,
    ListCtor,
    TupleCtor,
    /// `sum(v)` on a list: the reduction of §5.2 over a composite.
    BuiltinSum,
    /// `min(a, b, ...)` with several arguments, or `min(v)` on a composite.
    BuiltinMinMax {
        max: bool,
    },
    BuiltinPow,
    Complex,
    /// `print`, and every other call with no shape effect.
    NoShapeEffect,
    /// `copy.deepcopy`, `.copy()`: the argument's descriptor.
    Copy,
    Property(Prop),
    ListMethod(ListMethod),
    PromoteToDynamic,
}

/// A bare-name call: `coverage.md` §5.7's builtins (`src/ir_gen/named_attr/builtins.rs`).
pub fn builtin(member: &str) -> Option<Class> {
    Some(match member {
        "len" => Class::Len,
        "range" => Class::Range,
        "enumerate" => Class::Enumerate,
        "list" => Class::ListCtor,
        "tuple" => Class::TupleCtor,
        "sum" => Class::BuiltinSum,
        "any" => Class::Reduce(ReduceKind::Any),
        "all" => Class::Reduce(ReduceKind::All),
        "min" => Class::BuiltinMinMax { max: false },
        "max" => Class::BuiltinMinMax { max: true },
        "abs" => Class::Map(MapOut::RealPart, Domain::All),
        "int" => Class::Cast(Some(Dtype::I64)),
        "float" => Class::Cast(Some(Dtype::Float)),
        "bool" => Class::Cast(Some(Dtype::Bool)),
        "complex" => Class::Complex,
        "pow" => Class::BuiltinPow,
        "print" => Class::NoShapeEffect,
        _ => return None,
    })
}

/// `np.*` and `zinnia.*` (`src/ir_gen/named_attr/numpy_funcs.rs`,
/// `src/ops/registry.rs`).
pub fn numpy(member: &str) -> Option<Class> {
    if let Some(class) = numpy_elementwise(member) {
        return Some(class);
    }
    if let Some(class) = numpy_shape(member) {
        return Some(class);
    }
    numpy_other(member)
}

/// The elementwise set of `coverage.md` §5.1.
fn numpy_elementwise(member: &str) -> Option<Class> {
    let map = |out| Some(Class::Map(out, Domain::All));
    let zip = |op| Some(Class::Zip(op));
    match member {
        // unary, real result, with a domain
        "sqrt" => Some(Class::Map(MapOut::Float, Domain::NonNegative)),
        "log" => Some(Class::Map(MapOut::Float, Domain::Positive)),
        "arccos" | "acos" | "arcsin" | "asin" => {
            Some(Class::Map(MapOut::Float, Domain::UnitInterval))
        }
        // unary, real result
        "exp" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "arctan" | "atan"
        | "reciprocal" | "inv" => map(MapOut::Float),
        // unary, dtype preserved
        "negative" | "positive" | "sign" | "square" | "floor" | "ceil" | "trunc" | "round"
        | "deg2rad" | "radians" | "rad2deg" | "degrees" | "conj" | "conjugate" => map(MapOut::Same),
        "abs" | "absolute" | "fabs" | "real" | "imag" => map(MapOut::RealPart),
        "logical_not" => map(MapOut::Bool),
        // binary
        "add" => zip(ZipOp::Add),
        "subtract" => zip(ZipOp::Sub),
        "multiply" => zip(ZipOp::Mul),
        "divide" | "true_divide" => zip(ZipOp::Div),
        "floor_divide" => zip(ZipOp::FloorDiv),
        "mod" | "fmod" | "remainder" => zip(ZipOp::Mod),
        "power" | "pow" => zip(ZipOp::Pow),
        "equal" => zip(ZipOp::Eq),
        "not_equal" => zip(ZipOp::Ne),
        "less" => zip(ZipOp::Lt),
        "less_equal" => zip(ZipOp::Le),
        "greater" => zip(ZipOp::Gt),
        "greater_equal" => zip(ZipOp::Ge),
        "logical_and" => zip(ZipOp::And),
        "logical_or" => zip(ZipOp::Or),
        "logical_xor" => zip(ZipOp::Xor),
        "minimum" | "fmin" => zip(ZipOp::Min),
        "maximum" | "fmax" => zip(ZipOp::Max),
        "arctan2" | "atan2" => zip(ZipOp::Atan2),
        "clip" => Some(Class::Clip),
        "isclose" => Some(Class::IsClose),
        "allclose" | "array_equal" => Some(Class::Predicate),
        // the ufunc `.outer` forms the front end rewrites (`coverage.md` §2)
        "add_outer" | "subtract_outer" | "multiply_outer" | "divide_outer" => Some(Class::Outer),
        _ => None,
    }
}

/// The shape transforms of `coverage.md` §5.3 and the constructors of §5.4.
fn numpy_shape(member: &str) -> Option<Class> {
    Some(match member {
        "reshape" => Class::Reshape,
        "ravel" | "flatten" => Class::Ravel,
        "transpose" => Class::Transpose,
        "swapaxes" => Class::Swapaxes,
        "moveaxis" => Class::Moveaxis,
        "squeeze" => Class::Squeeze,
        "expand_dims" => Class::ExpandDims,
        "broadcast_to" => Class::BroadcastTo,
        "atleast_1d" => Class::AtLeast(1),
        "atleast_2d" => Class::AtLeast(2),
        "atleast_3d" => Class::AtLeast(3),
        "flip" | "flipud" | "fliplr" => Class::Reverse,
        "rot90" => Class::Rot90,
        "concatenate" => Class::Concat,
        "stack" => Class::Stack,
        "vstack" | "row_stack" => Class::StackFamily(StackKind::Vertical),
        "hstack" => Class::StackFamily(StackKind::Horizontal),
        "dstack" => Class::StackFamily(StackKind::Depth),
        "column_stack" => Class::StackFamily(StackKind::Column),
        "diag" => Class::Diag,
        "delete" => Class::Delete,
        "insert" => Class::Insert,
        "pad" => Class::Pad,
        "repeat" => Class::Repeat,
        "tile" => Class::Tile,
        "diff" => Class::Diff,
        "sort" => Class::Sort {
            index_result: false,
        },
        "argsort" => Class::Sort { index_result: true },
        "unique" => Class::Unique,
        "zeros" | "ones" | "empty" | "ndarray" => Class::Zeros,
        "full" => Class::Full,
        "zeros_like" | "ones_like" | "empty_like" | "full_like" => Class::Like,
        "identity" | "eye" => Class::Identity,
        "arange" => Class::Arange,
        "linspace" => Class::Linspace,
        "array" | "asarray" | "copy" => Class::Asarray,
        "indices" => Class::Indices,
        _ => return None,
    })
}

/// Reductions, gather and linear algebra, plus the dtype objects of §5.1.
fn numpy_other(member: &str) -> Option<Class> {
    if let Some(dtype) = dtype_object(member) {
        return Some(Class::DtypeClass(dtype));
    }
    Some(match member {
        "sum" => Class::Reduce(ReduceKind::Sum),
        "prod" => Class::Reduce(ReduceKind::Prod),
        "min" | "amin" => Class::Reduce(ReduceKind::Min),
        "max" | "amax" => Class::Reduce(ReduceKind::Max),
        "any" => Class::Reduce(ReduceKind::Any),
        "all" => Class::Reduce(ReduceKind::All),
        "mean" => Class::Reduce(ReduceKind::Mean),
        "var" => Class::Reduce(ReduceKind::Var),
        "std" => Class::Reduce(ReduceKind::Std),
        "argmax" => Class::Reduce(ReduceKind::ArgMax),
        "argmin" => Class::Reduce(ReduceKind::ArgMin),
        "cumsum" | "cumprod" => Class::Cumulative,
        "percentile" => Class::Percentile,
        "take" => Class::Take,
        "take_along_axis" => Class::TakeAlongAxis,
        "where" => Class::Where,
        "nonzero" => Class::Nonzero,
        "argwhere" => Class::Argwhere,
        "flatnonzero" => Class::Flatnonzero,
        "in1d" | "isin" => Class::In1d,
        "bincount" => Class::Bincount,
        "unravel_index" => Class::UnravelIndex,
        "ravel_multi_index" => Class::RavelMultiIndex,
        "dot" | "matmul" | "inner" => Class::Matmul,
        "tensordot" => Class::Tensordot,
        "einsum" => Class::Einsum,
        "corrcoef" => Class::Corrcoef,
        "correlate" => Class::Correlate,
        "promote_to_dynamic" => Class::PromoteToDynamic,
        "pi" | "e" => Class::FloatConstant,
        _ => return None,
    })
}

/// The `np.<dtype>` names, which are values of the baseline's `Value::Class`
/// kind (`coverage.md` §5.1). Every width collapses, as in the baseline.
fn dtype_object(member: &str) -> Option<Dtype> {
    Some(match member {
        "float16" | "float32" | "float64" | "double" | "single" | "half" | "longdouble" => {
            Dtype::Float
        }
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
        | "intp" | "uintp" | "long" | "byte" | "short" | "intc" | "uintc" | "ubyte" | "ushort"
        | "longlong" | "ulonglong" => Dtype::I64,
        "bool_" => Dtype::Bool,
        "complex64" | "complex128" | "complex256" | "csingle" | "cdouble" | "clongdouble" => {
            Dtype::Complex
        }
        _ => return None,
    })
}

/// `math.*` (`src/ops/registry.rs`'s `build_math_op`, plus the constants the
/// baseline's `np` fallback shares with it).
pub fn math(member: &str) -> Option<Class> {
    Some(match member {
        "sqrt" => Class::Map(MapOut::Float, Domain::NonNegative),
        "log" => Class::Map(MapOut::Float, Domain::Positive),
        "acos" | "asin" => Class::Map(MapOut::Float, Domain::UnitInterval),
        "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "exp" | "inv" => {
            Class::Map(MapOut::Float, Domain::All)
        }
        "fabs" => Class::Map(MapOut::RealPart, Domain::All),
        "atan2" => Class::Zip(ZipOp::Atan2),
        "floor" | "ceil" => Class::Map(MapOut::Same, Domain::All),
        "pi" | "e" => Class::FloatConstant,
        _ => return None,
    })
}

/// The array methods and properties of `coverage.md` §5.2, §5.3 and §5.8
/// (`src/ir_gen/named_attr/ndarray_methods.rs`), and the list methods of
/// §5.7 (`src/ir_gen/named_attr/list_methods.rs`).
///
/// The baseline tries the list table before the array table, so a name in
/// both (`copy`) is a list method; this follows the same order.
pub fn method(member: &str) -> Option<Class> {
    if let Some(class) = list_method(member) {
        return Some(class);
    }
    Some(match member {
        "shape" => Class::Property(Prop::Shape),
        "ndim" => Class::Property(Prop::Ndim),
        "size" => Class::Property(Prop::Size),
        "dtype" => Class::Property(Prop::Dtype),
        "T" => Class::Property(Prop::Transposed),
        "flat" => Class::Property(Prop::Flat),
        "real" | "conjugate" | "conj" => Class::Property(Prop::RealPart),
        "imag" => Class::Property(Prop::ImagPart),
        "astype" => Class::Cast(None),
        "tolist" => Class::ToList,
        "filter" => Class::Filter,
        // Everything else is the `np.*` name applied to the receiver as the
        // first argument, which is how the baseline's method table is built.
        other => return numpy(other),
    })
}

/// The list methods of `S-ListMethod` (§5.11).
fn list_method(member: &str) -> Option<Class> {
    Some(Class::ListMethod(match member {
        "append" => ListMethod::Append,
        "extend" => ListMethod::Extend,
        "insert" => ListMethod::Insert,
        "pop" => ListMethod::Pop,
        "remove" => ListMethod::Remove,
        "clear" => ListMethod::Clear,
        "reverse" => ListMethod::Reverse,
        "copy" => ListMethod::Copy,
        "count" => ListMethod::Count,
        "index" => ListMethod::Index,
        _ => return None,
    }))
}

/// `np.linalg.*` (`src/ir_gen/named_attr/mod.rs`'s `try_np_linalg_expr_attr`).
pub fn linalg(member: &str) -> Option<Class> {
    Some(match member {
        "norm" => Class::LinalgNorm,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `np.*` name `coverage.md` §8 counts in the parity corpus.
    const CORPUS_NUMPY: [&str; 54] = [
        "reshape",
        "sum",
        "array",
        "argsort",
        "arange",
        "unravel_index",
        "diag",
        "flatnonzero",
        "isclose",
        "take_along_axis",
        "zeros",
        "fliplr",
        "vstack",
        "ravel_multi_index",
        "logical_and",
        "floor",
        "empty_like",
        "sin",
        "deg2rad",
        "mean",
        "cumsum",
        "in1d",
        "indices",
        "delete",
        "insert",
        "sort",
        "empty",
        "argwhere",
        "zeros_like",
        "abs",
        "ones",
        "real",
        "imag",
        "where",
        "unique",
        "percentile",
        "min",
        "cos",
        "degrees",
        "arcsin",
        "max",
        "prod",
        "maximum",
        "minimum",
        "sqrt",
        "array_equal",
        "einsum",
        "tensordot",
        "correlate",
        "hstack",
        "corrcoef",
        "add_outer",
        "dot",
        "argmax",
    ];

    #[test]
    fn every_corpus_numpy_name_has_a_class() {
        for name in CORPUS_NUMPY {
            assert!(numpy(name).is_some(), "np.{} has no shape class", name);
        }
    }

    #[test]
    fn every_corpus_builtin_and_math_name_has_a_class() {
        for name in [
            "range",
            "len",
            "abs",
            "complex",
            "min",
            "float",
            "max",
            "list",
            "int",
            "enumerate",
            "sum",
            "any",
            "all",
        ] {
            assert!(
                builtin(name).is_some(),
                "builtin {} has no shape class",
                name
            );
        }
        for name in [
            "sqrt", "exp", "cos", "sin", "fabs", "pi", "log", "acos", "ceil",
        ] {
            assert!(math(name).is_some(), "math.{} has no shape class", name);
        }
    }

    #[test]
    fn list_methods_win_over_array_methods() {
        // The baseline tries `try_list_method` before `try_ndarray_method`.
        assert_eq!(
            method("copy"),
            Some(Class::ListMethod(ListMethod::Copy)),
            "`copy` must resolve to the list method, as in the baseline"
        );
        assert_eq!(
            method("insert"),
            Some(Class::ListMethod(ListMethod::Insert))
        );
    }

    #[test]
    fn a_method_name_falls_through_to_the_numpy_table() {
        assert_eq!(method("sum"), Some(Class::Reduce(ReduceKind::Sum)));
        assert_eq!(method("reshape"), Some(Class::Reshape));
    }
}
