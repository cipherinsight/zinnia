//! Fixed constructions (measurement only, `docs/ablations.md`).
//!
//! `ZINNIA_FORCE_CONSTRUCTION` is a comma-separated list of construction
//! names ([`Construction::name`]) and named configurations (`@name`,
//! [`CONFIGURATIONS`], expanded in place). For every group, the first listed
//! construction that *applies* to it, that is, whose candidates produce
//! every live lane of the group, is the only construction offering nodes for
//! that group (together with its helpers, [`HELPERS`]). A group to which no
//! listed construction applies is lowered as usual, and so is a group whose
//! forced construction cannot be selected together with the rest of the
//! program (`emit::lower_with` releases it). A forced construction is sound
//! like any other; it may only cost more. The benchmark harness never sets
//! the variable.

use super::construction::{Candidate, Construction};
use super::graph::{Group, Requirement};

/// The constructions forced on the groups they apply to, in order of
/// preference.
#[derive(Clone, Debug, Default)]
pub struct Forcing {
    names: Vec<String>,
    /// The families the named configurations fix (for the layer report:
    /// a group of such a family that keeps the layer's choice is a
    /// fallback).
    families: Vec<&'static str>,
}

/// The named configurations: one fixed construction per family (by
/// `OpKind::family`), whatever it costs; where a family lists two, they are
/// the same choice on different targets. A family left out keeps the
/// layer's choice.
///
/// * `delegate`: every access at a runtime index goes through the
///   backend's memory. On Noir the array is a native array, read through
///   bb's ROM and written through its RAM, so Noir's compiler and bb lower
///   the access instead of Zinnia; on halo2-base the counterpart is the
///   dynamic-lookup table read and the RAM trace. Only the families that
///   decide how an array is represented and accessed have such a
///   construction (element reads and the memory groups of written arrays).
///   The other families keep the layer's choice: their alternatives differ
///   in computing a result against checking it (sort, searchsorted, linear,
///   is_permutation), or have no memory-based alternative (wrap, scan,
///   window, permutation scatter, insert, mask write), or no choice at all
///   (corrcoef: the input type decides).
/// * `selects`: the simplest uniform lowering, select chains and plain
///   compute: element reads by a select chain over the index's equality
///   bits, written arrays as an overlay of selects, NumPy's index wrap by a
///   pinned sign and a select, scans by a select chain, sorts by a network
///   of compare-and-swaps, searchsorted by counting, matrix products and
///   correlations by their product sums, permutation checks by hit counts,
///   masked writes by selects. Insert (two forms of the same position
///   one-hot, neither a select) and corrcoef keep the layer's choice.
pub const CONFIGURATIONS: &[(&str, &[(&str, &[&str])])] = &[
    (
        "delegate",
        &[
            ("element", &["memory: element by table read"]),
            ("memory", &["memory: RAM trace", "memory: native RAM"]),
        ],
    ),
    (
        "selects",
        &[
            ("element", &["gather: element by a select chain"]),
            ("memory", &["memory: overlay of selects"]),
            ("wrap", &["memory: wrap by pinned sign"]),
            ("scan", &["gather: scan by a select chain"]),
            ("sort", &["sort: network"]),
            ("searchsorted", &["searchsorted: count"]),
            ("linear", &["linear: direct product sums"]),
            ("is_permutation", &["permutation: hit counts"]),
            ("mask write", &["mask write: selects"]),
        ],
    ),
];

/// Constructions that are a part of another one rather than an alternative
/// to it: forcing the first keeps the second. The multiplexer reads the
/// index bits a separate node produces (shared by the reads at one index).
const HELPERS: &[(&str, &str)] = &[("memory: element by index bits", "memory: index bits")];

impl Forcing {
    /// The forcing `ZINNIA_FORCE_CONSTRUCTION` asks for (none when unset).
    pub fn from_env() -> Forcing {
        Forcing::parse(&std::env::var("ZINNIA_FORCE_CONSTRUCTION").unwrap_or_default())
    }

    /// The forcing of a comma-separated list of names and `@configuration`s.
    ///
    /// # Panics
    /// On a name that is no registered construction and on an unknown
    /// configuration: a typo must not measure the default silently.
    pub fn parse(spec: &str) -> Forcing {
        let known: Vec<&str> = super::registry().iter().map(|c| c.name()).collect();
        let mut forcing = Forcing::default();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item.strip_prefix('@') {
                Some(configuration) => {
                    let Some((_, families)) = CONFIGURATIONS.iter().find(|(name, _)| *name == configuration) else {
                        panic!("ZINNIA_FORCE_CONSTRUCTION: no configuration @{configuration}");
                    };
                    for (family, listed) in *families {
                        forcing.families.push(family);
                        forcing.names.extend(listed.iter().map(|s| s.to_string()));
                    }
                }
                None => {
                    assert!(known.contains(&item), "ZINNIA_FORCE_CONSTRUCTION: no construction named {item:?}");
                    forcing.names.push(item.to_string());
                }
            }
        }
        forcing
    }

    /// A forcing of the given construction names, unchecked (tests force
    /// constructions of their own).
    pub fn of(names: &[&str]) -> Forcing {
        Forcing { names: names.iter().map(|s| s.to_string()).collect(), families: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The construction names, configurations expanded.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Whether a named configuration fixes the construction of `family`.
    pub fn fixes_family(&self, family: &str) -> bool {
        self.families.contains(&family)
    }

    /// The construction forced on `group`: the first listed one whose
    /// candidates (among `offered`) produce every live lane of the group.
    pub fn choice(&self, group: &Group, offered: &[(&dyn Construction, Vec<Candidate>)]) -> Option<&'static str> {
        self.names.iter().find_map(|name| {
            let (construction, candidates) = offered.iter().find(|(c, _)| c.name() == name.as_str())?;
            covers(group, candidates).then_some(construction.name())
        })
    }

    /// Whether `construction` keeps offering nodes for a group whose forced
    /// construction is `forced`.
    pub fn keeps(forced: &str, construction: &str) -> bool {
        construction == forced || HELPERS.contains(&(forced, construction))
    }
}

/// Whether `candidates` (some) produce every live lane of `group`.
fn covers(group: &Group, candidates: &[Candidate]) -> bool {
    !candidates.is_empty()
        && group.lanes.iter().all(|lane| candidates.iter().any(|c| c.post.contains(&Requirement::Lane(lane.stmt))))
}
