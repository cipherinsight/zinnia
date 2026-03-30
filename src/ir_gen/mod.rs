//! IR Generator — visitor pattern over Zinnia AST, produces IRGraph.
//! Ports `zinnia/compile/ir/ir_gen.py` (854 lines).

use std::collections::HashMap;

use crate::ast::*;
use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_ctx::IRContext;
use crate::types::Value;

mod visitors;
mod named_attr;
mod list_methods;
mod chip_external;
mod helpers;

#[cfg(test)]
mod tests;

/// Check if an assignment target node has the `star` flag set.
pub(super) fn is_starred_target(node: &ASTNode) -> bool {
    match node {
        ASTNode::ASTNameAssignTarget(t) => t.star,
        ASTNode::ASTSubscriptAssignTarget(t) => t.star,
        ASTNode::ASTStarredExpr(_) => true,
        _ => false,
    }
}

// Re-export SliceIndex from types for backward compatibility
pub use crate::types::SliceIndex;

/// Configuration for the IR generator.
#[derive(Debug, Clone)]
pub struct IRGenConfig {
    pub loop_limit: u32,
    pub recursion_limit: u32,
}

impl Default for IRGenConfig {
    fn default() -> Self {
        Self {
            loop_limit: 256,
            recursion_limit: 16,
        }
    }
}

/// Registered chip information for the IR generator.
#[derive(Debug, Clone)]
pub struct RegisteredChip {
    pub chip_ast: serde_json::Value,  // The ASTChip dict
    pub return_dt: serde_json::Value, // Return type descriptor
}

/// Registered external function information for the IR generator.
#[derive(Debug, Clone)]
pub struct RegisteredExternal {
    pub return_dt: serde_json::Value, // Return type descriptor
}

/// The IR Generator — walks the AST and produces an IRGraph.
pub struct IRGenerator {
    pub builder: IRBuilder,
    pub ctx: IRContext,
    config: IRGenConfig,
    registered_chips: HashMap<String, RegisteredChip>,
    registered_externals: HashMap<String, RegisteredExternal>,
    recursion_depth: u32,
    next_external_store_idx: u32,
}

impl IRGenerator {
    pub fn new(config: IRGenConfig) -> Self {
        Self {
            builder: IRBuilder::new(),
            ctx: IRContext::new(),
            config,
            registered_chips: HashMap::new(),
            registered_externals: HashMap::new(),
            recursion_depth: 0,
            next_external_store_idx: 1,
        }
    }

    /// Main entry point: generate an IRGraph from an AST circuit.
    pub fn generate(mut self, ast: &ASTCircuit) -> IRGraph {
        self.visit_circuit(ast);
        self.builder.export_ir_graph()
    }

    /// Generate from a JSON string (the bridge entry point).
    pub fn generate_from_json(config: IRGenConfig, ast_json: &str) -> Result<IRGraph, String> {
        let node: ASTNode = serde_json::from_str(ast_json)
            .map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let gen = IRGenerator::new(config);
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    /// Generate from JSON with chips and externals.
    pub fn generate_from_json_with_chips(
        config: IRGenConfig,
        ast_json: &str,
        chips: &HashMap<String, serde_json::Value>,
        externals: &HashMap<String, serde_json::Value>,
    ) -> Result<IRGraph, String> {
        let node: ASTNode = serde_json::from_str(ast_json)
            .map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let mut gen = IRGenerator::new(config);
                // Register chips
                for (name, chip_data) in chips {
                    gen.registered_chips.insert(name.clone(), RegisteredChip {
                        chip_ast: chip_data["chip_ast"].clone(),
                        return_dt: chip_data["return_dt"].clone(),
                    });
                }
                // Register externals
                for (name, ext_data) in externals {
                    gen.registered_externals.insert(name.clone(), RegisteredExternal {
                        return_dt: ext_data["return_dt"].clone(),
                    });
                }
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    // ── Visitor dispatch ──────────────────────────────────────────────

    fn visit(&mut self, node: &ASTNode) -> Value {
        match node {
            ASTNode::ASTCircuit(n) => { self.visit_circuit(n); Value::None }
            ASTNode::ASTAssignStatement(n) => { self.visit_assign(n); Value::None }
            ASTNode::ASTAugAssignStatement(n) => { self.visit_aug_assign(n); Value::None }
            ASTNode::ASTCondStatement(n) => { self.visit_cond(n); Value::None }
            ASTNode::ASTForInStatement(n) => { self.visit_for_in(n); Value::None }
            ASTNode::ASTWhileStatement(n) => { self.visit_while(n); Value::None }
            ASTNode::ASTBreakStatement(_) => { self.visit_break(); Value::None }
            ASTNode::ASTContinueStatement(_) => { self.visit_continue(); Value::None }
            ASTNode::ASTReturnStatement(n) => { self.visit_return(n); Value::None }
            ASTNode::ASTAssertStatement(n) => { self.visit_assert(n); Value::None }
            ASTNode::ASTPassStatement(_) => Value::None,
            ASTNode::ASTExprStatement(n) => self.visit(&n.expr),
            ASTNode::ASTBinaryOperator(n) => self.visit_binary_op(n),
            ASTNode::ASTUnaryOperator(n) => self.visit_unary_op(n),
            ASTNode::ASTNamedAttribute(n) => self.visit_named_attr(n),
            ASTNode::ASTExprAttribute(n) => self.visit_expr_attr(n),
            ASTNode::ASTLoad(n) => self.visit_load(n),
            ASTNode::ASTSubscriptExp(n) => self.visit_subscript(n),
            ASTNode::ASTConstantFloat(n) => self.builder.ir_constant_float(n.value),
            ASTNode::ASTConstantInteger(n) => self.builder.ir_constant_int(n.value),
            ASTNode::ASTConstantBoolean(n) => self.builder.ir_constant_bool(n.value),
            ASTNode::ASTConstantNone(_) => Value::None,
            ASTNode::ASTConstantString(n) => self.builder.ir_constant_str(n.value.clone()),
            ASTNode::ASTSquareBrackets(n) => self.visit_square_brackets(n),
            ASTNode::ASTParenthesis(n) => self.visit_parenthesis(n),
            ASTNode::ASTGeneratorExp(n) => self.visit_generator_exp(n),
            ASTNode::ASTCondExp(n) => self.visit_cond_exp(n),
            ASTNode::ASTJoinedStr(n) => self.visit_joined_str(n),
            ASTNode::ASTFormattedValue(n) => self.visit_formatted_value(n),
            ASTNode::ASTStarredExpr(_) => panic!("Can't use starred expression here"),
            // Assignment targets are not visited as expressions
            ASTNode::ASTNameAssignTarget(_)
            | ASTNode::ASTSubscriptAssignTarget(_)
            | ASTNode::ASTTupleAssignTarget(_)
            | ASTNode::ASTListAssignTarget(_)
            | ASTNode::ASTChip(_) => {
                panic!("Cannot visit {:?} as expression", std::mem::discriminant(node))
            }
        }
    }

    // ── Circuit ───────────────────────────────────────────────────────

    fn visit_circuit(&mut self, n: &ASTCircuit) {
        // Process inputs
        for (_i, inp) in n.inputs.iter().enumerate() {
            let dt = self.parse_dt_descriptor(&inp.annotation.dt);
            let kind = inp.annotation.kind.as_deref().unwrap_or("Private");
            let is_public = kind == "Public";
            let val = self.read_input_value(&dt, &inp.name, vec![], is_public);
            self.ctx.set(&inp.name, val);
        }

        self.register_global_datatypes();

        for stmt in &n.block {
            self.visit(stmt);
        }
    }

    // ── Statements ────────────────────────────────────────────────────
}
