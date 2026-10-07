//! The layer's view of the IR's array operations (`crate::ir_array`).
//!
//! A read of an `ArrayInit` reads known cells: in SSA its value is that cell
//! whatever other versions of the array exist, so the layer lowers it as an
//! **element read** of the cells its access can name (an `OpKind::Element`
//! group, read by the gather and memory constructions). Reads of written
//! versions belong to the version tree's **memory group**.

use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::types::StmtId;

/// An `ArrayRead` of an `ArrayInit`, as an element read: `line` are the
/// cells the access can name, in position order, and `index` the position
/// (`None` for a constant address: the read is `line[0]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementRead {
    pub line: Vec<StmtId>,
    pub index: Option<StmtId>,
}

/// `stmt` as an element read, when it is an `ArrayRead` of an `ArrayInit`
/// with at most one runtime position. `stmts` holds every statement by id.
///
/// # Panics
/// On a read of an `ArrayInit` at several runtime positions: the frontend
/// and the layer's emitters read known cells along one combined index.
pub fn element_read(stmts: &[IRStatement], stmt: StmtId) -> Option<ElementRead> {
    let read = stmts.get(stmt as usize)?;
    let IR::ArrayRead(access) = &read.ir else { return None };
    let version = &stmts[read.arguments[0] as usize];
    if !matches!(version.ir, IR::ArrayInit { .. }) {
        return None;
    }
    assert!(access.rank() <= 1, "a read of known cells at {} runtime positions: {}", access.rank(), access.describe());
    let line = access.line().into_iter().map(|a| version.arguments[a]).collect();
    Some(ElementRead { line, index: read.arguments.get(1).copied() })
}
