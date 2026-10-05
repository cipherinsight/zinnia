//! The rule library: one module per rule file of
//! `revision/phase1/rules/`, under the same name.
//!
//! A rule is a template (`../../README.md` §5.3); an implementation node is
//! the template with the source values, the input interfaces and every
//! integer filled in. [`Rule::build`] is the whole of a rule file's
//! `source_pattern`, `required_input_ports`, `premises`,
//! `resource_relations` and `emitter`: it matches, asks its premises through
//! the oracle, fills the row of each input port, and returns one
//! [`Candidate`] per combination of concrete child requirements, with its
//! parameters and its cost.
//!
//! The library is the 44 rules of `rules/README.md` §11: the nine scalar
//! rules of step 1; the 22 fixed-shape array rules of step 2 —
//! `BUILD_DENSE`; `MAKE_REF`, `MAKE_REF_RUNTIME`, `REINDEX_REF`;
//! `ENUM_COORDS`, `APPLY_INDEX_MAP`, `GATHER`, `READ_DIRECT`,
//! `READ_KEY_MUX`, `READ_ROM`, `ROM_BIND`, `ROM_CLOSE_EXACT`; `MAP_ALIGNED`,
//! `ELEMENTWISE_BATCH`, `BROADCAST_MAP`; `EXPAND_FIBERS`, `FOLD_GROUPS`,
//! `MATMUL`, `CUMSUM`; `INDEX_NORMALIZE`, `SHAPE_CHECK`; `CONSTANT_SIZED` —
//! and the 11 rules of step 3, which handle a value whose length is decided
//! at run time: `PREFIX_RANK`, `COUNT`, `NONZERO`; `BUILD_MASKED`,
//! `PACK_BY_KEY`, `PACK_SEGMENTS`, `CONCAT`, `SETITEM`; `SORT`, `ARGSORT`,
//! `UNIQUE`; and the two the coverage pass added, `DELETE` and `INSERT`,
//! which move one line of one axis.

#[cfg(test)]
pub(crate) mod cycle_probe;
pub(crate) mod elementwise;
pub(crate) mod mapping;
pub(crate) mod masks;
pub(crate) mod ordering;
pub(crate) mod reduction;
pub(crate) mod representation;
pub(crate) mod reading;
pub(crate) mod scalar_control;
pub(crate) mod sources;

use crate::phase0::desc::ValueKey;
use crate::phase0::term::BoolId;

use crate::phase1::ast_index::{AstNode, Constant};
use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::{Parameter, Target};
use crate::phase1::interface::{Batch, Family, Interface};
use crate::phase1::oracle::answers::AnswerId;
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

/// The names phase 0's walker treats as modules rather than as a receiver
/// (`../../phase0/walk/calls.rs`, `named_attribute`).
const MODULES: [&str; 7] =
    ["np", "numpy", "zinnia", "math", "cmath", "copy", "DynamicNDArray"];

/// One constant parameter of a call on an array, counted from the array.
///
/// A rule file writes every call with the array first — `swapaxes(u, x, y)`
/// and `squeeze(u, axis=k)` of `mapping/make_ref.md`, `take(a, I, axis=k)` of
/// `reading/gather.md` — and `at` is the position in that spelling. Python
/// writes the same call two ways: `np.swapaxes(a, x, y)`, where the array is
/// argument 0 and `x` is argument 1, and `a.swapaxes(x, y)`, where the array
/// is the receiver and `x` is argument 0. The AST index keeps the two apart
/// (`../ast_index.rs`): an `ASTExprAttribute` is a `MethodCall`, and
/// `a.swapaxes(…)` written on a name is a `Call` whose target is that name
/// and not a module. A keyword argument is spelled the same in both and is
/// looked up by name first, which is [`AstNode::parameter`]'s own order.
pub fn parameter<'a>(node: &'a AstNode, at: usize, name: &str) -> Option<&'a Constant> {
    let receiver = match node {
        AstNode::MethodCall { .. } => true,
        AstNode::Call { target: Some(target), .. } => !MODULES.contains(&target.as_str()),
        _ => false,
    };
    node.parameter(at.saturating_sub(usize::from(receiver)), name)
}

/// The rule names of `rules/README.md` §11. The rules of steps 1 and 2 have
/// a module; the rest are named so that a "no provider" report can say which
/// rule is missing rather than that nothing matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RuleId {
    InputStore,
    InputScalar,
    Constant,
    ConstantSized,
    BuildDense,
    BuildMasked,
    PackByKey,
    PackSegments,
    Concat,
    SetItem,
    Delete,
    Insert,
    MakeRef,
    MakeRefRuntime,
    ReindexRef,
    EnumCoords,
    ApplyIndexMap,
    Gather,
    ReadDirect,
    ReadKeyMux,
    ReadRom,
    RomBind,
    RomCloseExact,
    MapAligned,
    ElementwiseBatch,
    BroadcastMap,
    ExpandFibers,
    FoldGroups,
    Matmul,
    Cumsum,
    Sort,
    Argsort,
    Unique,
    PrefixRank,
    Count,
    Nonzero,
    ScalarOp,
    LenShape,
    TermToWires,
    Select,
    IndexNormalize,
    ShapeCheck,
    EnforceCheck,
    Export,
    /// The test-only rule of [`cycle_probe`], which is in the library only
    /// under `cfg(test)`.
    #[cfg(test)]
    CycleProbe,
}

impl RuleId {
    pub fn as_str(self) -> &'static str {
        match self {
            RuleId::InputStore => "INPUT_STORE",
            RuleId::InputScalar => "INPUT_SCALAR",
            RuleId::Constant => "CONSTANT",
            RuleId::ConstantSized => "CONSTANT_SIZED",
            RuleId::BuildDense => "BUILD_DENSE",
            RuleId::BuildMasked => "BUILD_MASKED",
            RuleId::PackByKey => "PACK_BY_KEY",
            RuleId::PackSegments => "PACK_SEGMENTS",
            RuleId::Concat => "CONCAT",
            RuleId::SetItem => "SETITEM",
            RuleId::Delete => "DELETE",
            RuleId::Insert => "INSERT",
            RuleId::MakeRef => "MAKE_REF",
            RuleId::MakeRefRuntime => "MAKE_REF_RUNTIME",
            RuleId::ReindexRef => "REINDEX_REF",
            RuleId::EnumCoords => "ENUM_COORDS",
            RuleId::ApplyIndexMap => "APPLY_INDEX_MAP",
            RuleId::Gather => "GATHER",
            RuleId::ReadDirect => "READ_DIRECT",
            RuleId::ReadKeyMux => "READ_KEY_MUX",
            RuleId::ReadRom => "READ_ROM",
            RuleId::RomBind => "ROM_BIND",
            RuleId::RomCloseExact => "ROM_CLOSE_EXACT",
            RuleId::MapAligned => "MAP_ALIGNED",
            RuleId::ElementwiseBatch => "ELEMENTWISE_BATCH",
            RuleId::BroadcastMap => "BROADCAST_MAP",
            RuleId::ExpandFibers => "EXPAND_FIBERS",
            RuleId::FoldGroups => "FOLD_GROUPS",
            RuleId::Matmul => "MATMUL",
            RuleId::Cumsum => "CUMSUM",
            RuleId::Sort => "SORT",
            RuleId::Argsort => "ARGSORT",
            RuleId::Unique => "UNIQUE",
            RuleId::PrefixRank => "PREFIX_RANK",
            RuleId::Count => "COUNT",
            RuleId::Nonzero => "NONZERO",
            RuleId::ScalarOp => "SCALAR_OP",
            RuleId::LenShape => "LEN_SHAPE",
            RuleId::TermToWires => "TERM_TO_WIRES",
            RuleId::Select => "SELECT",
            RuleId::IndexNormalize => "INDEX_NORMALIZE",
            RuleId::ShapeCheck => "SHAPE_CHECK",
            RuleId::EnforceCheck => "ENFORCE_CHECK",
            RuleId::Export => "EXPORT",
            #[cfg(test)]
            RuleId::CycleProbe => "CYCLE_PROBE",
        }
    }
}

/// One output bundle of a candidate: the requirement it is attached to.
///
/// A node has several when they are projections of one set of wires
/// (`rules/README.md` §3.4): `INPUT_STORE.box` answers a `Store` and a
/// `Masked` from one validity computation.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub target: Target,
    pub interface: Interface,
    pub condition: BoolId,
    /// The batch, when the requirement this bundle answers is a
    /// `Fetch(v, batch, g)` and not a `Represent` (`../README.md` §2). The
    /// two forms are one output list, because a rule that answers a `Fetch`
    /// may answer a `Represent` from the same wires (§3.4).
    pub batch: Option<Batch>,
}

/// One implementation node a rule offers, before it is attached to G.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub rule: RuleId,
    pub variant: Option<&'static str>,
    pub matched: Vec<ValueKey>,
    pub inputs: Vec<crate::phase1::graph::ReqId>,
    pub outputs: Vec<OutputSpec>,
    pub parameters: Vec<(String, Parameter)>,
    pub epoch: u32,
    /// The primitives of `rules/README.md` §7 the emitter counts, which the
    /// cost function turns into a number.
    pub primitives: Vec<(Primitive, i128)>,
    pub answers: Vec<AnswerId>,
    /// The candidate readers of a ROM context, with the queries each would
    /// register (`reading/rom_close_exact.md`, `resource_relations`): phase 2
    /// receives `Q_o = Σ_r x_r·Q_r` over their selection variables. Empty on
    /// every other node.
    pub readers: Vec<(crate::phase1::graph::ReqId, i128)>,
}

impl Candidate {
    pub fn new(rule: RuleId, variant: Option<&'static str>, epoch: u32) -> Candidate {
        Candidate {
            rule,
            variant,
            matched: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters: Vec::new(),
            epoch,
            primitives: Vec::new(),
            answers: Vec::new(),
            readers: Vec::new(),
        }
    }

    pub fn output(mut self, target: Target, interface: Interface, condition: BoolId) -> Candidate {
        self.outputs.push(OutputSpec { target, interface, condition, batch: None });
        self
    }

    /// The bundle that answers a `Fetch(v, batch, g)` (§3.3).
    pub fn answers_fetch(
        mut self,
        target: Target,
        batch: Batch,
        interface: Interface,
        condition: BoolId,
    ) -> Candidate {
        self.outputs.push(OutputSpec {
            target,
            interface,
            condition,
            batch: Some(batch),
        });
        self
    }

    pub fn parameter(mut self, name: &str, value: i128) -> Candidate {
        self.parameters.push((name.to_string(), Parameter::Fixed(value)));
        self
    }

    /// The one parameter `../../README.md` §5.7 admits as a range: phase 2
    /// receives it as an integer variable bounded by `lo` and `hi`.
    pub fn ranged(mut self, name: &str, lo: i128, hi: i128) -> Candidate {
        self.parameters.push((name.to_string(), Parameter::Ranged { lo, hi }));
        self
    }

    pub fn emit(mut self, primitive: Primitive, times: i128) -> Candidate {
        if times > 0 {
            self.primitives.push((primitive, times));
        }
        self
    }
}

/// One rule of the library.
pub(crate) trait Rule: Sync {
    fn id(&self) -> RuleId;

    /// The families the rule's `produced_output_bundle` can answer, so that
    /// a row skips the rules that cannot fill it.
    fn answers(&self) -> &'static [Family];

    /// Match, ask the premises, fill the input rows, and offer the
    /// implementation nodes.
    fn build(&self, builder: &mut Builder<'_>, request: &RowRequest)
        -> Result<Vec<Candidate>, Reject>;
}

static INPUT_SCALAR: sources::input_scalar::InputScalar = sources::input_scalar::InputScalar;
static INPUT_STORE: sources::input_store::InputStore = sources::input_store::InputStore;
static CONSTANT: sources::constant::Constant = sources::constant::Constant;
static SCALAR_OP: scalar_control::scalar_op::ScalarOp = scalar_control::scalar_op::ScalarOp;
static LEN_SHAPE: scalar_control::len_shape::LenShape = scalar_control::len_shape::LenShape;
static TERM_TO_WIRES: scalar_control::term_to_wires::TermToWires =
    scalar_control::term_to_wires::TermToWires;
static SELECT: scalar_control::select::Select = scalar_control::select::Select;
static ENFORCE_CHECK: scalar_control::enforce_check::EnforceCheck =
    scalar_control::enforce_check::EnforceCheck;
static EXPORT: scalar_control::export::Export = scalar_control::export::Export;
#[cfg(test)]
static CYCLE_PROBE: cycle_probe::CycleProbe = cycle_probe::CycleProbe;
static MAKE_REF: mapping::make_ref::MakeRef = mapping::make_ref::MakeRef;
static MAKE_REF_RUNTIME: mapping::make_ref_runtime::MakeRefRuntime =
    mapping::make_ref_runtime::MakeRefRuntime;
static REINDEX_REF: mapping::reindex_ref::ReindexRef = mapping::reindex_ref::ReindexRef;
static CONSTANT_SIZED: sources::constant_sized::ConstantSized =
    sources::constant_sized::ConstantSized;
static EXPAND_FIBERS: reduction::expand_fibers::ExpandFibers =
    reduction::expand_fibers::ExpandFibers;
static FOLD_GROUPS: reduction::fold_groups::FoldGroups = reduction::fold_groups::FoldGroups;
static MATMUL: reduction::matmul::Matmul = reduction::matmul::Matmul;
static CUMSUM: reduction::cumsum::Cumsum = reduction::cumsum::Cumsum;
static INDEX_NORMALIZE: scalar_control::index_normalize::IndexNormalize =
    scalar_control::index_normalize::IndexNormalize;
static SHAPE_CHECK: scalar_control::shape_check::ShapeCheck =
    scalar_control::shape_check::ShapeCheck;
static BUILD_DENSE: representation::build_dense::BuildDense =
    representation::build_dense::BuildDense;
static MAP_ALIGNED: elementwise::map_aligned::MapAligned = elementwise::map_aligned::MapAligned;
static ELEMENTWISE_BATCH: elementwise::elementwise_batch::ElementwiseBatch =
    elementwise::elementwise_batch::ElementwiseBatch;
static BROADCAST_MAP: elementwise::broadcast_map::BroadcastMap =
    elementwise::broadcast_map::BroadcastMap;
static ENUM_COORDS: reading::enum_coords::EnumCoords = reading::enum_coords::EnumCoords;
static APPLY_INDEX_MAP: reading::apply_index_map::ApplyIndexMap =
    reading::apply_index_map::ApplyIndexMap;
static GATHER: reading::gather::Gather = reading::gather::Gather;
static READ_DIRECT: reading::read_direct::ReadDirect = reading::read_direct::ReadDirect;
static READ_KEY_MUX: reading::read_key_mux::ReadKeyMux = reading::read_key_mux::ReadKeyMux;
static READ_ROM: reading::read_rom::ReadRom = reading::read_rom::ReadRom;
static ROM_BIND: reading::rom_bind::RomBind = reading::rom_bind::RomBind;
static ROM_CLOSE_EXACT: reading::rom_close_exact::RomCloseExact =
    reading::rom_close_exact::RomCloseExact;
static PREFIX_RANK: masks::prefix_rank::PrefixRank = masks::prefix_rank::PrefixRank;
static COUNT: masks::count::Count = masks::count::Count;
static NONZERO: masks::nonzero::Nonzero = masks::nonzero::Nonzero;
static BUILD_MASKED: representation::build_masked::BuildMasked =
    representation::build_masked::BuildMasked;
static PACK_BY_KEY: representation::pack_by_key::PackByKey =
    representation::pack_by_key::PackByKey;
static PACK_SEGMENTS: representation::pack_segments::PackSegments =
    representation::pack_segments::PackSegments;
static CONCAT: representation::concat::Concat = representation::concat::Concat;
static SETITEM: representation::setitem::SetItem = representation::setitem::SetItem;
static DELETE: representation::delete::Delete = representation::delete::Delete;
static INSERT: representation::insert::Insert = representation::insert::Insert;
static SORT: ordering::sort::Sort = ordering::sort::Sort;
static ARGSORT: ordering::argsort::Argsort = ordering::argsort::Argsort;
static UNIQUE: ordering::unique::Unique = ordering::unique::Unique;

/// The rules of the library that implement one transfer rule of phase 0
/// (`rules/README.md` §9, "Coverage of phase 0").
///
/// A value whose row has no provider is reported with this, so that a
/// rejection names the rule step 1 is missing rather than saying only that
/// nothing matched.
pub(crate) fn covering_rules(rule: crate::phase0::rules::Rule) -> &'static [RuleId] {
    use crate::phase0::rules::Rule as P0;
    match rule {
        P0::EInputFixed | P0::EInputVarLen | P0::EInputSymbolic | P0::EInputComposite
        | P0::EInputHashed => &[RuleId::InputStore, RuleId::InputScalar],
        P0::EConst | P0::EArithShape | P0::EArithData | P0::ECmp => &[RuleId::ScalarOp],
        P0::EZeros | P0::EOnes | P0::EArange => &[RuleId::Constant, RuleId::ConstantSized],
        P0::ELen | P0::EShape | P0::EProperty => &[RuleId::LenShape],
        P0::EMap | P0::EZip | P0::EWhere3 => {
            &[RuleId::MapAligned, RuleId::ElementwiseBatch, RuleId::BroadcastMap]
        }
        P0::EIndex | P0::ESlice | P0::ENewAxis | P0::ETranspose | P0::EMoveaxis
        | P0::EReshape | P0::ESqueeze | P0::EBroadcast | P0::EReverse | P0::EDiag => {
            &[RuleId::MakeRef, RuleId::MakeRefRuntime, RuleId::ReindexRef]
        }
        P0::EElem | P0::ETake | P0::ETakeBroadcast => {
            &[RuleId::Gather, RuleId::ReadKeyMux, RuleId::ReadRom]
        }
        P0::EFilter1D | P0::EFilterAxis0 | P0::EFilterSame => {
            &[RuleId::BuildMasked, RuleId::PackByKey, RuleId::PrefixRank, RuleId::Count]
        }
        P0::ENonzero => &[RuleId::Nonzero],
        P0::EConcat | P0::EStack => &[RuleId::Concat, RuleId::PackSegments],
        P0::SSetItem => &[RuleId::SetItem],
        P0::EDelete => &[RuleId::Delete],
        P0::EInsert => &[RuleId::Insert],
        P0::EReduceScalar | P0::EReduceAxis => &[RuleId::ExpandFibers, RuleId::FoldGroups],
        P0::EMatmul => &[RuleId::Matmul],
        P0::ECumsum => &[RuleId::Cumsum],
        P0::ECorrelate => &[RuleId::Matmul],
        P0::ESort => &[RuleId::Sort, RuleId::Argsort],
        P0::EUnique => &[RuleId::Unique],
        P0::EValueSized => &[RuleId::ConstantSized],
        P0::MergeArray | P0::MergeScalar | P0::SIf | P0::ECondExp => &[RuleId::Select],
        P0::EComp => &[RuleId::PackByKey],
        // Every other rule of §5 is either no rule of this library's
        // (aliases, unrolling, keys) or one `rules/README.md` §9 lists as
        // not covered.
        _ => &[],
    }
}

/// The library. A later step appends; the engine does not change.
pub(crate) const RULES: &[&(dyn Rule + Sync)] = &[
    &INPUT_SCALAR,
    &INPUT_STORE,
    &CONSTANT,
    &SCALAR_OP,
    &LEN_SHAPE,
    &TERM_TO_WIRES,
    &SELECT,
    &CONSTANT_SIZED,
    &BUILD_DENSE,
    &EXPAND_FIBERS,
    &FOLD_GROUPS,
    &MATMUL,
    &CUMSUM,
    &INDEX_NORMALIZE,
    &SHAPE_CHECK,
    &MAP_ALIGNED,
    &ELEMENTWISE_BATCH,
    &BROADCAST_MAP,
    &MAKE_REF,
    &MAKE_REF_RUNTIME,
    &REINDEX_REF,
    &ENUM_COORDS,
    &APPLY_INDEX_MAP,
    &GATHER,
    &READ_DIRECT,
    &READ_KEY_MUX,
    &READ_ROM,
    &ROM_BIND,
    &ROM_CLOSE_EXACT,
    &PREFIX_RANK,
    &COUNT,
    &NONZERO,
    &BUILD_MASKED,
    &PACK_BY_KEY,
    &PACK_SEGMENTS,
    &CONCAT,
    &SETITEM,
    &DELETE,
    &INSERT,
    &SORT,
    &ARGSORT,
    &UNIQUE,
    &ENFORCE_CHECK,
    &EXPORT,
    #[cfg(test)]
    &CYCLE_PROBE,
];
