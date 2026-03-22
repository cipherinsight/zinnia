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
            let dict = ty.to_dt_dict();
            let restored = ZinniaType::from_dt_dict(&dict)
                .unwrap_or_else(|e| panic!("Failed to round-trip {:?}: {}", ty, e));
            assert_eq!(ty, restored, "Round-trip failed for {:?}", ty);
        }
    }

    #[test]
    fn test_value_creation() {
        let v = Value::Integer(ScalarValue::known(42, 0));
        assert_eq!(v.zinnia_type(), ZinniaType::Integer);
        assert_eq!(v.ptr(), Some(0));
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
        assert_eq!(v.ptr(), Option::None);
    }

    #[test]
    fn test_number_type() {
        assert_eq!(ZinniaType::Integer.number_type(), Some(NumberType::Integer));
        assert_eq!(ZinniaType::Float.number_type(), Some(NumberType::Float));
        assert_eq!(ZinniaType::Boolean.number_type(), Some(NumberType::Integer));
        assert_eq!(ZinniaType::String.number_type(), Option::None);
    }
}
