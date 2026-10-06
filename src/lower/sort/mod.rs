//! The sort family: `np.sort`, `np.argsort`, `np.median`, `np.percentile`
//! (`OpKind::Sort`).

mod semantics;

pub use semantics::{SortSemantics, INDICES, VALUES};

use super::Construction;

/// The sort family's constructions.
pub fn constructions() -> Vec<&'static dyn Construction> {
    Vec::new()
}
