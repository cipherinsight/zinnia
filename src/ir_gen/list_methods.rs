use crate::types::{CompositeData, Value};

use super::IRGenerator;

impl IRGenerator {
    pub(crate) fn list_method_append(&mut self, var: &str, args: &[Value]) -> Value {
        if let Some(new_elem) = args.first() {
            if let Some(Value::List(mut data)) = self.ctx.get(var) {
                data.elements_type.push(new_elem.zinnia_type());
                data.values.push(new_elem.clone());
                self.ctx.set(var, Value::List(data));
            }
        }
        Value::None
    }

    pub(crate) fn list_method_extend(&mut self, var: &str, args: &[Value]) -> Value {
        if let Some(other) = args.first() {
            if let (Some(Value::List(mut data)), Value::List(ext) | Value::Tuple(ext)) =
                (self.ctx.get(var), other)
            {
                data.elements_type.extend(ext.elements_type.clone());
                data.values.extend(ext.values.clone());
                self.ctx.set(var, Value::List(data));
            }
        }
        Value::None
    }

    pub(crate) fn list_method_pop(&mut self, var: &str, args: &[Value]) -> Value {
        // Try to get static index from args
        let idx_val = args.first();
        // Only use static path if: (a) no args (default -1), or (b) arg is compile-time known
        let static_idx = match idx_val {
            None => Some(-1i64),  // default: pop last
            Some(v) => v.int_val(),  // None if dynamic
        };

        if let Some(idx) = static_idx {
            if let Some(Value::List(mut data)) = self.ctx.get(var) {
                let len = data.values.len() as i64;
                let i = if idx < 0 { len + idx } else { idx };
                if i < 0 || i >= len {
                    panic!("pop index out of range");
                }
                let i = i as usize;
                let removed = data.values.remove(i);
                data.elements_type.remove(i);
                self.ctx.set(var, Value::List(data));
                return removed;
            }
        }
        // Dynamic index case: select the popped value and rebuild list without it
        if let Some(idx_v) = idx_val {
            if let Some(Value::List(data)) = self.ctx.get(var) {
                let n = data.values.len();
                let len_const = self.builder.ir_constant_int(n as i64);
                let neg_len = self.builder.ir_constant_int(-(n as i64));

                // Normalize negative index: idx = idx < 0 ? idx + len : idx
                let zero = self.builder.ir_constant_int(0);
                let is_neg = self.builder.ir_less_than_i(idx_v, &zero);
                let normalized = self.builder.ir_add_i(idx_v, &len_const);
                let idx_norm = self.builder.ir_select_i(&is_neg, &normalized, idx_v);

                // Assert 0 <= idx_norm < len
                let ge_zero = self.builder.ir_greater_than_or_equal_i(&idx_norm, &zero);
                let lt_len = self.builder.ir_less_than_i(&idx_norm, &len_const);
                let in_bounds = self.builder.ir_logical_and(&ge_zero, &lt_len);
                self.builder.ir_assert(&in_bounds);

                let popped = crate::helpers::value_ops::dynamic_list_subscript(&mut self.builder, &data, &idx_norm);
                if n > 0 {
                    let mut past_idx = self.builder.ir_constant_bool(false);
                    let mut new_values = Vec::new();
                    let mut new_types = Vec::new();
                    for i in 0..n - 1 {
                        let i_const = self.builder.ir_constant_int(i as i64);
                        let is_idx = self.builder.ir_equal_i(&idx_norm, &i_const);
                        past_idx = self.builder.ir_logical_or(&past_idx, &is_idx);
                        // If past the popped index, take data[i+1], else data[i]
                        let shifted = self.builder.ir_select_i(&past_idx, &data.values[i + 1], &data.values[i]);
                        new_values.push(shifted);
                        new_types.push(data.elements_type[i].clone());
                    }
                    let new_list = Value::List(CompositeData { elements_type: new_types, values: new_values });
                    self.ctx.set(var, new_list);
                }
                return popped;
            }
        }
        Value::None
    }

    pub(crate) fn list_method_remove(&mut self, var: &str, args: &[Value]) -> Value {
        if let Some(target) = args.first() {
            if let Some(Value::List(data)) = self.ctx.get(var) {
                // Try static removal first
                let target_int = target.int_val();
                if let Some(target_val) = target_int {
                    let mut new_data = data.clone();
                    // Check if all elements have known values (static list)
                    let all_known = new_data.values.iter().all(|v| v.int_val().is_some());
                    if let Some(pos) = new_data.values.iter().position(|v| v.int_val() == Some(target_val)) {
                        new_data.values.remove(pos);
                        new_data.elements_type.remove(pos);
                        self.ctx.set(var, Value::List(new_data));
                        return Value::None;
                    } else if all_known {
                        // Static value not found in a fully known list
                        panic!("Value not found in list");
                    }
                }
                // Dynamic: generate a new list with the first matching element removed
                // Strategy: for each position, if we haven't removed yet and this matches,
                // skip it (use next element). Otherwise, keep current or shifted element.
                let n = data.values.len();
                let mut found = self.builder.ir_constant_bool(false);
                let mut new_values = Vec::new();
                let mut new_types = Vec::new();

                // Build shifted list: for each output position i,
                // if found_before_i: take data[i+1], else take data[i]
                for i in 0..n - 1 {
                    let eq = self.builder.ir_equal_i(&data.values[i], target);
                    let not_found = self.builder.ir_logical_not(&found);
                    let is_removal = self.builder.ir_logical_and(&eq, &not_found);
                    found = self.builder.ir_logical_or(&found, &is_removal);
                    // After this point, 'found' means we've removed an element at or before i
                    // If found, take data[i+1], else take data[i]
                    let shifted = self.builder.ir_select_i(&found, &data.values[i + 1], &data.values[i]);
                    new_values.push(shifted);
                    new_types.push(data.elements_type[i].clone());
                }
                // For the last element: if not found yet, check it too
                let eq_last = self.builder.ir_equal_i(&data.values[n - 1], target);
                let not_found_last = self.builder.ir_logical_not(&found);
                let is_removal_last = self.builder.ir_logical_and(&eq_last, &not_found_last);
                found = self.builder.ir_logical_or(&found, &is_removal_last);

                // Assert that the value was found
                self.builder.ir_assert(&found);

                // Update the list variable with the shorter list
                let new_list = Value::List(CompositeData { elements_type: new_types, values: new_values });
                self.ctx.set(var, new_list);
            }
        }
        Value::None
    }

    pub(crate) fn list_method_clear(&mut self, var: &str) -> Value {
        self.ctx.set(var, Value::List(CompositeData { elements_type: vec![], values: vec![] }));
        Value::None
    }

    pub(crate) fn list_method_reverse(&mut self, var: &str) -> Value {
        if let Some(Value::List(mut data)) = self.ctx.get(var) {
            data.values.reverse();
            data.elements_type.reverse();
            self.ctx.set(var, Value::List(data));
        }
        Value::None
    }

    pub(crate) fn list_method_insert(&mut self, var: &str, args: &[Value]) -> Value {
        assert!(args.len() >= 2, "list.insert requires index and object arguments");
        let idx_arg = &args[0];
        let new_elem = &args[1];

        if let Some(Value::List(mut data)) = self.ctx.get(var) {
            if let Some(idx) = idx_arg.int_val() {
                // Static index: insert at known position
                let len = data.values.len() as i64;
                let i = if idx < 0 {
                    (len + idx).max(0) as usize
                } else {
                    (idx as usize).min(data.values.len())
                };
                data.elements_type.insert(i, new_elem.zinnia_type());
                data.values.insert(i, new_elem.clone());
                self.ctx.set(var, Value::List(data));
            } else {
                // Dynamic index: build a select chain over all possible positions.
                // All elements must be the same type for the select chain to work.
                let len = data.values.len();
                // Build list with element inserted at each possible position 0..=len
                // and select the correct one based on the index value.
                let mut current_data = data.clone();
                // Start with insert at position `len` (append)
                current_data.elements_type.push(new_elem.zinnia_type());
                current_data.values.push(new_elem.clone());

                // For each position i from len-1 down to 0, conditionally swap
                for i in (0..len).rev() {
                    let i_const = self.builder.ir_constant_int(i as i64);
                    let should_insert_here = self.builder.ir_equal_i(idx_arg, &i_const);
                    // If inserting at i, shift elements right from position i
                    // Use select: for positions >= i, pick element from position-1 in original
                    let mut new_vals = current_data.values.clone();
                    // Swap: move new_elem to position i and shift others right
                    for j in (i + 1..=len).rev() {
                        new_vals[j] = self.builder.ir_select_i(&should_insert_here, &current_data.values[j - 1], &current_data.values[j]);
                    }
                    new_vals[i] = self.builder.ir_select_i(&should_insert_here, new_elem, &current_data.values[i]);
                    current_data.values = new_vals;
                }
                let types = current_data.values.iter().map(|v| v.zinnia_type()).collect();
                current_data.elements_type = types;
                self.ctx.set(var, Value::List(current_data));
            }
        }
        Value::None
    }

    pub(crate) fn list_method_count(&mut self, var: &str, args: &[Value]) -> Value {
        if let (Some(target), Some(Value::List(data) | Value::Tuple(data))) = (args.first(), self.ctx.get(var)) {
            // Generate dynamic count: sum(1 if elem == target else 0 for elem in list)
            let mut count = self.builder.ir_constant_int(0);
            for elem in &data.values {
                let eq = self.builder.ir_equal_i(elem, target);
                // Cast bool to int and add
                let one = self.builder.ir_constant_int(1);
                let zero = self.builder.ir_constant_int(0);
                let inc = self.builder.ir_select_i(&eq, &one, &zero);
                count = self.builder.ir_add_i(&count, &inc);
            }
            count
        } else {
            self.builder.ir_constant_int(0)
        }
    }

    pub(crate) fn list_method_index(&mut self, var: &str, args: &[Value]) -> Value {
        if let (Some(target), Some(Value::List(data) | Value::Tuple(data))) = (args.first(), self.ctx.get(var)) {
            let start = args.get(1).and_then(|a| a.int_val()).unwrap_or(0) as usize;
            // Generate dynamic index: find first match after start
            let mut found = self.builder.ir_constant_bool(false);
            let mut answer = self.builder.ir_constant_int(0);
            for (i, elem) in data.values.iter().enumerate() {
                if i < start { continue; }
                let eq = self.builder.ir_equal_i(elem, target);
                let not_found = self.builder.ir_logical_not(&found);
                let first_match = self.builder.ir_logical_and(&eq, &not_found);
                let idx_const = self.builder.ir_constant_int(i as i64);
                answer = self.builder.ir_select_i(&first_match, &idx_const, &answer);
                found = self.builder.ir_logical_or(&found, &eq);
            }
            // Assert found (the element should be in the list for valid circuits)
            self.builder.ir_assert(&found);
            answer
        } else {
            self.builder.ir_constant_int(-1)
        }
    }
}
