//! Ω, the candidate policy of `revision/phase1/rules/README.md` §6.4.
//!
//! Ω bounds the compiler's search, never the program's meaning
//! (`../README.md` §5.8): how deep a `Ref` may nest, how many recomputation
//! instances one value may have, which discretised resource values are
//! admitted, how large G may grow. It also carries the width policy of §6.1
//! and the two backend switches the library reads by name.

/// The candidate policy and the width policy, with the defaults phase 1 uses
/// when a caller supplies none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Omega {
    /// Bound on `Ref.span` (§6.4).
    pub map_depth: u32,
    /// Instances of one value under distinct sharing identities.
    pub recompute: u32,
    /// Nodes of G.
    pub node_budget: usize,
    /// `PACK_BY_KEY` and `NONZERO` network.
    pub routing: Routing,
    /// Every width is rounded up into this ladder and bounded by
    /// [`Omega::max_width`] (§6.1).
    pub widths: Vec<u32>,
    pub max_width: u32,
    /// Zero the tail of an exported `Packed` (default on).
    pub export_canonical: bool,
    /// Enforce dtype ranges of public inputs.
    pub range_check_public: bool,
    /// Wrap once per fold when the unwrapped sum fits the field.
    pub field_accumulate: bool,
    /// How many later-epoch variants one rule may instantiate for a value
    /// (`rules/README.md` §6.2).
    pub epoch_variants: u32,
    /// The largest capacity a MAX may allocate, per axis and in the product
    /// (§6.4, "admitted values of discretized resource parameters").
    ///
    /// A MAX over a term whose only bound is a dtype range answers with that
    /// range — `(len(a) div n)·n` for an unconstrained `n` is bounded only by
    /// 2^63 — and a rule that allocated it would ask for a payload no
    /// machine holds. The capacity is a resource parameter like any other, so
    /// Ω says which values are admitted: above this one a rule offers no
    /// node and the row says so, which is a rejection naming the capacity
    /// rather than a compiler that runs out of memory.
    pub max_capacity: i128,
}

/// `Ω.routing` of §6.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routing {
    Select,
    Permute,
}

impl Default for Omega {
    fn default() -> Self {
        Omega {
            map_depth: 4,
            recompute: 1,
            node_budget: 200_000,
            routing: Routing::Select,
            // No rounding by default: the ladder is every width up to the
            // maximum, so `round` is the identity and a backend that wants
            // a coarser ladder sets one.
            widths: Vec::new(),
            max_width: 254,
            export_canonical: true,
            range_check_public: false,
            field_accumulate: false,
            epoch_variants: 2,
            max_capacity: 1 << 20,
        }
    }
}

impl Omega {
    /// A width rounded up into `Ω.widths` (§6.1). An empty ladder admits
    /// every width.
    pub fn round(&self, width: u32) -> u32 {
        match self.widths.iter().copied().filter(|&w| w >= width).min() {
            Some(rounded) => rounded,
            None => width,
        }
    }

    /// Whether a width is admitted at all (§6.1, `Ω.max_width`).
    pub fn admits(&self, width: u32) -> bool {
        width <= self.max_width
    }
}
