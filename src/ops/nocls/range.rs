use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct RangeOp;

impl RangeOp {
    const PARAMS: [ParamEntry; 3] = [
        ParamEntry::required("start"),
        ParamEntry::optional("stop"),
        ParamEntry::optional("step"),
    ];
}

impl Op for RangeOp {
    fn name(&self) -> &'static str { "range" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let start_val = args.require("start");
        let stop_opt = args.get("stop");
        let step_opt = args.get("step");

        // All range arguments must be statically known integers.
        let (start, stop, step) = if let Some(stop_v) = stop_opt {
            let s = start_val.int_val().expect("range: start must be a constant integer");
            let e = stop_v.int_val().expect("range: stop must be a constant integer");
            let st = step_opt
                .and_then(|v| v.int_val())
                .unwrap_or(1);
            (s, e, st)
        } else {
            // range(stop) — start=0, stop=start_val
            let e = start_val.int_val().expect("range: stop must be a constant integer");
            (0, e, 1)
        };

        assert!(step != 0, "range: step must not be zero");

        let mut values = Vec::new();
        let mut types = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            values.push(builder.ir_constant_int(i));
            types.push(crate::types::ZinniaType::Integer);
            i += step;
        }

        Value::List(crate::types::CompositeData {
            elements_type: types,
            values,
        })
    }
}
