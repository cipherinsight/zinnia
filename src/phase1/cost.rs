//! The cost function of `revision/phase1/README.md` §1 and the unit cost of
//! `rules/README.md` §7.
//!
//! An implementation node's cost is `cost(emitter, parameters)`, an integer
//! in the backend's units. The backend's cost function does not exist yet,
//! so phase 1 ships [`UnitCost`]: the primitive counts of §7 with weight 1
//! per primitive. The trait is what lets the backend replace it without
//! touching a rule.
//!
//! # Float
//!
//! A Float wire holds the backend's encoding, not a field element with a
//! width, so no `range_w`, no `wrap_w` and no decomposition is emitted for
//! one. Every Float operation is one [`FloatOp`], named after the backend's
//! own instruction in `src/ir_defs.rs` so that phase 3 maps it one to one,
//! and a rule that needs several — `isclose`, `np.round` — emits several.

/// One float primitive, `fop⟨name⟩` of `rules/README.md` §7.
///
/// One variant per instruction of the backend's float vocabulary
/// (`src/ir_defs.rs`: "Float arithmetic", "Float comparison", "Math
/// functions (float)", `FloatCast`, `SelectF`, `ConstantFloat`), and
/// [`FloatOp::as_str`] gives that instruction's own name, so a rule states
/// which backend gadget its row denotes and phase 3 needs no table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatOp {
    // Arithmetic.
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    /// A float base raised to an integer exponent (`PowFI`).
    PowInt,
    Abs,
    Sign,
    /// The sum aggregate over `len` operands (`SumF`).
    Sum,
    /// The product-sum aggregate over `len` pairs (`ProductSumF`).
    ProductSum,
    // Comparison.
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
    // Math functions.
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    SinH,
    CosH,
    TanH,
    ArcSin,
    ArcCos,
    ArcTan2,
    // Selection, casts and literals.
    Select,
    /// An integer or Boolean wire read as a float (`FloatCast`).
    FromInt,
    /// A float wire read as an integer (`IntCast`).
    ToInt,
    /// A float literal on a wire (`ConstantFloat`).
    Constant,
}

impl FloatOp {
    /// The backend instruction this row denotes, spelled as
    /// `IR::to_string` spells it (`src/ir_defs.rs`).
    pub fn as_str(self) -> &'static str {
        match self {
            FloatOp::Add => "add_f",
            FloatOp::Sub => "sub_f",
            FloatOp::Mul => "mul_f",
            FloatOp::Div => "div_f",
            FloatOp::FloorDiv => "floor_divide_f",
            FloatOp::Mod => "mod_f",
            FloatOp::Pow => "pow_f",
            FloatOp::PowInt => "pow_f_int",
            FloatOp::Abs => "abs_f",
            FloatOp::Sign => "sign_f",
            FloatOp::Sum => "sum_f",
            FloatOp::ProductSum => "product_sum_f",
            FloatOp::Eq => "eq_f",
            FloatOp::Ne => "ne_f",
            FloatOp::Lt => "lt_f",
            FloatOp::Lte => "lte_f",
            FloatOp::Gt => "gt_f",
            FloatOp::Gte => "gte_f",
            FloatOp::Sqrt => "sqrt_f",
            FloatOp::Exp => "exp_f",
            FloatOp::Log => "log_f",
            FloatOp::Sin => "sin_f",
            FloatOp::Cos => "cos_f",
            FloatOp::Tan => "tan_f",
            FloatOp::SinH => "sinh_f",
            FloatOp::CosH => "cosh_f",
            FloatOp::TanH => "tanh_f",
            FloatOp::ArcSin => "arcsin_f",
            FloatOp::ArcCos => "arccos_f",
            FloatOp::ArcTan2 => "arctan2_f",
            FloatOp::Select => "select_f",
            FloatOp::FromInt => "float_cast",
            FloatOp::ToInt => "int_cast",
            FloatOp::Constant => "constant_float",
        }
    }
}

/// One primitive of `rules/README.md` §7, with the parameters its count
/// needs. `w` is a bit width, `B` a capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    /// A linear relation among wires and constants: 0.
    Lin,
    Mul,
    Bool,
    And,
    Or,
    Xor,
    /// `1 − b`: 0.
    Not,
    Select,
    Eq,
    /// `cmp_w`: `w + 1`.
    Cmp { width: u32 },
    /// `range_w`: `w`.
    Range { width: u32 },
    /// `divmod_k` with a constant `k`: `range_w(q) + range_{⌈log2 k⌉}(r)`.
    DivModConst { quotient_width: u32, divisor_width: u32, power_of_two: bool },
    /// `divmod_d` by a wire: `1 + cmp_w + range_w`.
    DivModWire { width: u32 },
    /// `wrap_w`: `range_w` plus a carry range check.
    Wrap { width: u32, carry_width: u32 },
    /// `therm(n over B)`: `2B − 1`.
    Therm { capacity: i128 },
    /// `onehot(k over B, γ)`: `B + 1`.
    OneHot { capacity: i128 },
    /// `perm(B, k)`: the backend's `P(B, k)`.
    Perm { capacity: i128, arity: u32 },
    /// Membership in a table: the backend's.
    Lookup,
    /// `fop⟨name⟩`, one float primitive of `rules/README.md` §7.
    Float(FloatOp),
    /// A hash of `n` cells: the backend's.
    Hash { cells: i128 },
    /// One materialised cell: the backend's.
    Cell,
    /// A prover-supplied wire: cells, not gates.
    Witness,
}

/// The cost function `../README.md` §1 takes as an argument.
///
/// `Sync` because the construction runs on a thread of its own, for the
/// stack it needs (`crate::phase1::build_with`).
pub trait Cost: Sync {
    /// The cost of one primitive occurrence, in the backend's units.
    fn primitive(&self, primitive: Primitive) -> i128;

    /// The cost of a whole emitter, given the primitives it emits.
    fn emitter(&self, primitives: &[(Primitive, i128)]) -> i128 {
        primitives
            .iter()
            .map(|&(primitive, times)| self.primitive(primitive) * times)
            .sum()
    }

    /// The step cost of a ROM context at a total query count
    /// (`../README.md` §1). No rule of step 1 builds one.
    fn rom_context(&self, _capacity: i128, _queries: i128) -> i128 {
        0
    }
}

/// The counts of `rules/README.md` §7 with weight 1 per primitive.
///
/// Where §7 says "backend", the count is 0 here and the report says so:
/// `perm`, `lookup`, `hash` and `cell` have no unit until a backend supplies
/// one, and charging a made-up number would put a number phase 2 optimises
/// against into the graph. `fop⟨name⟩` is the one exception: §7 charges it
/// 1 unit, and the backend replaces that number with the cost of its own
/// fixed-point gadget, which differs per operation (a `sqrt_f` is not an
/// `add_f`). One is charged rather than zero because a zero would make
/// Float work free and phase 2 would select a Float route over an integer
/// one whatever the backend charges.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnitCost;

impl Cost for UnitCost {
    fn primitive(&self, primitive: Primitive) -> i128 {
        match primitive {
            Primitive::Lin | Primitive::Not | Primitive::Witness => 0,
            Primitive::Mul
            | Primitive::Bool
            | Primitive::And
            | Primitive::Or
            | Primitive::Xor
            | Primitive::Select => 1,
            Primitive::Eq => 2,
            Primitive::Cmp { width } => width as i128 + 1,
            Primitive::Range { width } => width as i128,
            Primitive::DivModConst { quotient_width, divisor_width, power_of_two } => {
                quotient_width as i128
                    + divisor_width as i128
                    + if power_of_two { 0 } else { divisor_width as i128 + 1 }
            }
            Primitive::DivModWire { width } => 1 + (width as i128 + 1) + width as i128,
            Primitive::Wrap { width, carry_width } => width as i128 + carry_width as i128,
            Primitive::Therm { capacity } => (2 * capacity - 1).max(0),
            Primitive::OneHot { capacity } => capacity + 1,
            // `fop⟨name⟩` is charged 1 (§7): the number is the backend's
            // and it replaces it, but a zero would make phase 2 treat every
            // Float operation as free and pick the Float route whatever it
            // costs.
            Primitive::Float(_) => 1,
            // Backend units (§7): no number until the backend gives one.
            Primitive::Perm { .. }
            | Primitive::Lookup
            | Primitive::Hash { .. }
            | Primitive::Cell => 0,
        }
    }
}
