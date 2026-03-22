//! Zinnia AST types — deserialized from Python `export()` dicts via serde.
//! Ports `zinnia/compile/ast/` (46 node types).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DebugInfo
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugInfo {
    #[serde(default)]
    pub line: Option<u32>,
    #[serde(default)]
    pub col: Option<u32>,
    #[serde(default)]
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------
// Top-level tagged AST node — dispatched by `__class__`
// ---------------------------------------------------------------------------

/// A tagged AST node from Python's `export()` format.
/// The `__class__` field determines which variant to deserialize into.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "__class__")]
pub enum ASTNode {
    // ── Circuit / Chip ────────────────────────────────────────────
    ASTCircuit(ASTCircuit),
    ASTChip(ASTChip),

    // ── Statements ────────────────────────────────────────────────
    ASTAssignStatement(ASTAssignStatement),
    ASTAugAssignStatement(ASTAugAssignStatement),
    ASTCondStatement(ASTCondStatement),
    ASTForInStatement(ASTForInStatement),
    ASTWhileStatement(ASTWhileStatement),
    ASTBreakStatement(ASTBreakStatement),
    ASTContinueStatement(ASTContinueStatement),
    ASTReturnStatement(ASTReturnStatement),
    ASTAssertStatement(ASTAssertStatement),
    ASTPassStatement(ASTPassStatement),
    ASTExprStatement(ASTExprStatement),

    // ── Expressions ───────────────────────────────────────────────
    ASTBinaryOperator(ASTBinaryOperator),
    ASTUnaryOperator(ASTUnaryOperator),
    ASTNamedAttribute(ASTNamedAttribute),
    ASTExprAttribute(ASTExprAttribute),
    ASTLoad(ASTLoad),
    ASTSubscriptExp(ASTSubscriptExp),
    ASTConstantFloat(ASTConstantFloat),
    ASTConstantInteger(ASTConstantInteger),
    ASTConstantBoolean(ASTConstantBoolean),
    ASTConstantNone(ASTConstantNone),
    ASTConstantString(ASTConstantString),
    ASTSquareBrackets(ASTSquareBrackets),
    ASTParenthesis(ASTParenthesis),
    ASTGeneratorExp(ASTGeneratorExp),
    ASTCondExp(ASTCondExp),
    ASTJoinedStr(ASTJoinedStr),
    ASTFormattedValue(ASTFormattedValue),
    ASTStarredExpr(ASTStarredExpr),

    // ── Assignment targets ────────────────────────────────────────
    ASTNameAssignTarget(ASTNameAssignTarget),
    ASTSubscriptAssignTarget(ASTSubscriptAssignTarget),
    ASTTupleAssignTarget(ASTTupleAssignTarget),
    ASTListAssignTarget(ASTListAssignTarget),
}

// ---------------------------------------------------------------------------
// Circuit / Chip
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTCircuit {
    #[serde(default)]
    pub block: Vec<ASTNode>,
    #[serde(default)]
    pub inputs: Vec<ASTCircuitInput>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTCircuitInput {
    pub name: String,
    pub annotation: ASTAnnotation,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTAnnotation {
    pub dt: serde_json::Value, // DTDescriptor dict — deserialized on demand
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTChip {
    #[serde(default)]
    pub block: Vec<ASTNode>,
    #[serde(default)]
    pub inputs: Vec<ASTChipInput>,
    pub return_dt: serde_json::Value,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTChipInput {
    pub name: String,
    #[serde(default)]
    pub annotation: Option<ASTAnnotation>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTAssignStatement {
    pub targets: Vec<ASTNode>,
    pub value: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTAugAssignStatement {
    pub targets: Vec<ASTNode>,
    pub op_type: String,
    pub value: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTCondStatement {
    pub cond: Box<ASTNode>,
    pub t_block: Vec<ASTNode>,
    pub f_block: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTForInStatement {
    pub target: Box<ASTNode>,
    pub iter_expr: Box<ASTNode>,
    pub block: Vec<ASTNode>,
    #[serde(default)]
    pub orelse: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTWhileStatement {
    pub test_expr: Box<ASTNode>,
    pub block: Vec<ASTNode>,
    #[serde(default)]
    pub orelse: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTBreakStatement {
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTContinueStatement {
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTReturnStatement {
    #[serde(default)]
    pub expr: Option<Box<ASTNode>>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTAssertStatement {
    pub expr: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTPassStatement {
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTExprStatement {
    pub expr: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTBinaryOperator {
    pub operator: String,
    pub lhs: Box<ASTNode>,
    pub rhs: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTUnaryOperator {
    pub operator: String,
    pub operand: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTNamedAttribute {
    #[serde(default)]
    pub target: Option<String>,
    pub member: String,
    #[serde(default)]
    pub args: Vec<ASTNode>,
    #[serde(default)]
    pub kwargs: HashMap<String, ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTExprAttribute {
    pub target: Box<ASTNode>,
    pub member: String,
    #[serde(default)]
    pub args: Vec<ASTNode>,
    #[serde(default)]
    pub kwargs: HashMap<String, ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTLoad {
    pub name: String,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTSubscriptExp {
    pub val: Box<ASTNode>,
    pub slicing: ASTSlice,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTConstantFloat {
    pub value: f64,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTConstantInteger {
    pub value: i64,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTConstantBoolean {
    pub value: bool,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTConstantNone {
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTConstantString {
    pub value: String,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTSquareBrackets {
    pub values: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTParenthesis {
    pub values: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTGeneratorExp {
    pub elt: Box<ASTNode>,
    pub generators: Vec<ASTGenerator>,
    pub kind: String,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTGenerator {
    pub target: Box<ASTNode>,
    #[serde(rename = "iter")]
    pub iter_expr: Box<ASTNode>,
    #[serde(default)]
    pub ifs: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTCondExp {
    pub cond: Box<ASTNode>,
    pub t_expr: Box<ASTNode>,
    pub f_expr: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTJoinedStr {
    pub values: Vec<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTFormattedValue {
    pub value: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTStarredExpr {
    pub value: Box<ASTNode>,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

// ---------------------------------------------------------------------------
// Assignment targets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTNameAssignTarget {
    pub name: String,
    #[serde(default)]
    pub star: bool,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTSubscriptAssignTarget {
    pub target: Box<ASTNode>,
    pub slicing: ASTSlice,
    #[serde(default)]
    pub star: bool,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTTupleAssignTarget {
    pub targets: Vec<ASTNode>,
    #[serde(default)]
    pub star: bool,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTListAssignTarget {
    pub targets: Vec<ASTNode>,
    #[serde(default)]
    pub star: bool,
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTSlice {
    pub data: Vec<serde_json::Value>, // Each element: ASTNode dict or [start, stop, step] array
    #[serde(default)]
    pub dbg: Option<DebugInfo>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_constant_int() {
        let json = r#"{"__class__": "ASTConstantInteger", "value": 42}"#;
        let node: ASTNode = serde_json::from_str(json).unwrap();
        if let ASTNode::ASTConstantInteger(n) = node {
            assert_eq!(n.value, 42);
        } else {
            panic!("Expected ASTConstantInteger");
        }
    }

    #[test]
    fn test_deserialize_binary_op() {
        let json = r#"{
            "__class__": "ASTBinaryOperator",
            "operator": "add",
            "lhs": {"__class__": "ASTConstantInteger", "value": 1},
            "rhs": {"__class__": "ASTConstantInteger", "value": 2}
        }"#;
        let node: ASTNode = serde_json::from_str(json).unwrap();
        if let ASTNode::ASTBinaryOperator(n) = node {
            assert_eq!(n.operator, "add");
        } else {
            panic!("Expected ASTBinaryOperator");
        }
    }

    #[test]
    fn test_deserialize_circuit() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {"__class__": "ASTPassStatement"}
            ],
            "inputs": []
        }"#;
        let node: ASTNode = serde_json::from_str(json).unwrap();
        if let ASTNode::ASTCircuit(c) = node {
            assert_eq!(c.block.len(), 1);
            assert_eq!(c.inputs.len(), 0);
        } else {
            panic!("Expected ASTCircuit");
        }
    }

    #[test]
    fn test_deserialize_assign() {
        let json = r#"{
            "__class__": "ASTAssignStatement",
            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
            "value": {"__class__": "ASTConstantInteger", "value": 10}
        }"#;
        let node: ASTNode = serde_json::from_str(json).unwrap();
        if let ASTNode::ASTAssignStatement(a) = node {
            assert_eq!(a.targets.len(), 1);
        } else {
            panic!("Expected ASTAssignStatement");
        }
    }
}
