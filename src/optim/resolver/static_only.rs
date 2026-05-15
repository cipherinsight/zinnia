//! The default `Resolver`: no-op cache, delegates straight to the existing
//! `Value` accessors. Behaviourally identical to pre-P0 code.

use crate::types::Value;

use super::Resolver;

/// The default `Resolver`: no-op cache, delegates straight to the existing
/// `Value` accessors. Behaviourally identical to pre-P0 code.
#[derive(Debug, Default)]
pub struct StaticOnlyResolver;

impl StaticOnlyResolver {
    pub fn new() -> Self {
        Self
    }
}

impl Resolver for StaticOnlyResolver {
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }

    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        val.bool_val()
    }

    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        // For a fully-static integer, the value is its own max. P2's
        // `RangeResolver` will tighten this for non-literal expressions.
        val.int_val()
    }

    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }
}
