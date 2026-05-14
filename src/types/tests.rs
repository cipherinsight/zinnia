#[cfg(test)]
mod tests {
    use crate::types::*;

    #[test]
    fn test_zinnia_type_display() {
        assert_eq!(ZinniaType::Integer.to_string(), "Integer");
        assert_eq!(ZinniaType::Float.to_string(), "Float");
        assert_eq!(ZinniaType::Boolean.to_string(), "Boolean");
        assert_eq!(
            ZinniaType::NDArray {
                shape: vec![3, 4],
                dtype: NumberType::Float,
            }
            .to_string(),
            "NDArray[Float, 3, 4]"
        );
        assert_eq!(
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 100,
                max_rank: 2,
            }
            .to_string(),
            "DynamicNDArray[Integer, 100, 2]"
        );
        assert_eq!(
            ZinniaType::List {
                elements: vec![ZinniaType::Integer, ZinniaType::Float],
            }
            .to_string(),
            "List[Integer, Float]"
        );
        assert_eq!(
            ZinniaType::PoseidonHashed {
                dtype: Box::new(ZinniaType::Integer),
            }
            .to_string(),
            "PoseidonHashed[Integer]"
        );
    }

    #[test]
    fn test_type_predicates() {
        assert!(ZinniaType::Integer.is_number());
        assert!(ZinniaType::Float.is_number());
        assert!(ZinniaType::Boolean.is_number());
        assert!(!ZinniaType::String.is_number());

        assert!(ZinniaType::Integer.is_integer());
        assert!(ZinniaType::Boolean.is_integer());
        assert!(!ZinniaType::Float.is_integer());

        assert!(ZinniaType::NDArray {
            shape: vec![2],
            dtype: NumberType::Integer,
        }
        .is_ndarray());
        assert!(ZinniaType::DynamicNDArray {
            dtype: NumberType::Float,
            max_length: 10,
            max_rank: 1,
        }
        .is_ndarray());
    }

    #[test]
    fn test_implicit_cast() {
        assert!(ZinniaType::Boolean.can_implicit_cast_to(&ZinniaType::Integer));
        assert!(ZinniaType::Integer.can_implicit_cast_to(&ZinniaType::Integer));
        assert!(!ZinniaType::Integer.can_implicit_cast_to(&ZinniaType::Float));
        assert!(!ZinniaType::Float.can_implicit_cast_to(&ZinniaType::Integer));
    }

    #[test]
    fn test_from_annotation() {
        assert_eq!(
            ZinniaType::from_annotation("int", &[]).unwrap(),
            ZinniaType::Integer
        );
        assert_eq!(
            ZinniaType::from_annotation("Float", &[]).unwrap(),
            ZinniaType::Float
        );
        assert_eq!(
            ZinniaType::from_annotation("NDArray", &[
                AnnotationArg::Type(ZinniaType::Float),
                AnnotationArg::Int(3),
                AnnotationArg::Int(4),
            ])
            .unwrap(),
            ZinniaType::NDArray {
                shape: vec![3, 4],
                dtype: NumberType::Float,
            }
        );
        assert_eq!(
            ZinniaType::from_annotation("DynamicNDArray", &[
                AnnotationArg::Type(ZinniaType::Integer),
                AnnotationArg::Int(100),
                AnnotationArg::Int(2),
            ])
            .unwrap(),
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 100,
                max_rank: 2,
            }
        );
        assert!(ZinniaType::from_annotation("NDArray", &[]).is_err());
        assert!(ZinniaType::from_annotation("unknown_type", &[]).is_err());
    }

    #[test]
    fn test_num_elements() {
        assert_eq!(
            ZinniaType::NDArray {
                shape: vec![3, 4],
                dtype: NumberType::Float,
            }
            .num_elements(),
            Some(12)
        );
        assert_eq!(
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 100,
                max_rank: 2,
            }
            .num_elements(),
            Some(100)
        );
        assert_eq!(ZinniaType::Integer.num_elements(), Option::None);
    }

    #[test]
    fn test_dt_dict_round_trip() {
        let types = vec![
            ZinniaType::Integer,
            ZinniaType::Float,
            ZinniaType::Boolean,
            ZinniaType::String,
            ZinniaType::None,
            ZinniaType::Class,
            ZinniaType::NDArray {
                shape: vec![2, 3],
                dtype: NumberType::Float,
            },
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 50,
                max_rank: 3,
            },
            ZinniaType::List {
                elements: vec![ZinniaType::Integer, ZinniaType::Float],
            },
            ZinniaType::Tuple {
                elements: vec![ZinniaType::String, ZinniaType::Boolean],
            },
            ZinniaType::PoseidonHashed {
                dtype: Box::new(ZinniaType::Integer),
            },
        ];

        for ty in types {
            let json = serde_json::to_string(&ty)
                .unwrap_or_else(|e| panic!("Failed to serialize {:?}: {}", ty, e));
            let restored: ZinniaType = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("Failed to deserialize {:?}: {}", json, e));
            assert_eq!(ty, restored, "Round-trip failed for {:?}", ty);
        }
    }

    #[test]
    fn test_value_creation() {
        let v = Value::Integer(ScalarValue::known(42, 0));
        assert_eq!(v.zinnia_type(), ZinniaType::Integer);
        assert_eq!(v.stmt_id(), Some(0));
        assert_eq!(v.int_val(), Some(42));
        assert!(v.is_number());
        assert!(v.is_integer());

        let v = Value::Float(ScalarValue::known(3.14, 1));
        assert_eq!(v.zinnia_type(), ZinniaType::Float);
        assert_eq!(v.float_val(), Some(3.14));

        let v = Value::Boolean(ScalarValue::known(true, 2));
        assert_eq!(v.zinnia_type(), ZinniaType::Boolean);
        assert_eq!(v.bool_val(), Some(true));
        assert!(v.is_integer()); // Boolean is integer-like

        let v = Value::None;
        assert_eq!(v.zinnia_type(), ZinniaType::None);
        assert_eq!(v.stmt_id(), Option::None);
    }

    #[test]
    fn test_number_type() {
        assert_eq!(ZinniaType::Integer.number_type(), Some(NumberType::Integer));
        assert_eq!(ZinniaType::Float.number_type(), Some(NumberType::Float));
        assert_eq!(ZinniaType::Boolean.number_type(), Some(NumberType::Integer));
        assert_eq!(ZinniaType::String.number_type(), Option::None);
    }

    /// `Value::List` and `Value::Tuple` must surface their `CompositeData`'s
    /// `value_id`, so static composites can anchor facts the same way
    /// scalars and dyn ndarrays already do
    /// (compiler.value-list-tuple-value-id).
    #[test]
    fn test_value_list_has_value_id_for_fact_anchoring() {
        let list = Value::List(CompositeData::new(
            vec![ZinniaType::Integer],
            vec![Value::Integer(ScalarValue::constant(7))],
        ));
        let vid = list
            .value_id()
            .expect("Value::List should expose a value_id");

        // Two distinct constructions must yield distinct identities, even
        // when structurally identical or empty.
        let list2 = Value::List(CompositeData::new(vec![], vec![]));
        assert_ne!(vid, list2.value_id().unwrap());

        // Tuples are wired through the same arm.
        let tup = Value::Tuple(CompositeData::new(vec![], vec![]));
        assert!(tup.value_id().is_some());
        assert_ne!(tup.value_id().unwrap(), vid);
    }

    /// `Value::StaticArray` carries a `value_id` so fact-emission on the
    /// segment-native static-array path lands in `facts.per_value` and
    /// becomes provable downstream (compiler.value-static-array-value-id).
    #[test]
    fn fact_anchoring_works_on_static_array_value_id() {
        use crate::optim::predicates::formula::{ContractTerm, ContractVar};
        use crate::optim::prove::ProveOutcome;
        use std::collections::HashMap;

        let mut b = crate::builder::IRBuilder::new();
        let arr = Value::StaticArray {
            dtype: NumberType::Integer,
            shape: vec![3],
            segment_id: 0,
            strides: vec![1],
            offset: 0,
            imag_segment_id: None,
            value_id: ValueId::next(),
        };
        let vid = arr.value_id().expect("static array now has a value_id");

        b.fire_contract("zeros_content", vid, &HashMap::new());

        let q = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(vid)),
                ContractTerm::LitInt(0),
            ],
        };
        assert!(matches!(b.prove(&q), ProveOutcome::Proved));

        // Two distinct constructions must yield distinct identities.
        let arr2 = Value::StaticArray {
            dtype: NumberType::Integer,
            shape: vec![3],
            segment_id: 0,
            strides: vec![1],
            offset: 0,
            imag_segment_id: None,
            value_id: ValueId::next(),
        };
        assert_ne!(vid, arr2.value_id().unwrap());
    }
}
