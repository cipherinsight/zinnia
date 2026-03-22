#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_simple_constants() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                    "value": {"__class__": "ASTConstantInteger", "value": 42}
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        assert!(!graph.is_empty()); // At least the constant + type registrations
    }

    #[test]
    fn test_generate_binary_op() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "z"}],
                    "value": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "add",
                        "lhs": {"__class__": "ASTConstantInteger", "value": 10},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 20}
                    }
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        // Should have: constant(10), constant(20), add_i
        assert!(graph.len() >= 3);
    }

    #[test]
    fn test_generate_if_else() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "cond"}],
                    "value": {"__class__": "ASTConstantBoolean", "value": true}
                },
                {
                    "__class__": "ASTCondStatement",
                    "cond": {"__class__": "ASTLoad", "name": "cond"},
                    "t_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 1}
                        }
                    ],
                    "f_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 2}
                        }
                    ]
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        assert!(graph.len() >= 3);
    }
}
