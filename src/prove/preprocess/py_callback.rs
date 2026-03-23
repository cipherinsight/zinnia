//! PyO3 adapter: implements `ExternalCallback` by calling Python functions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::prove::error::ProvingError;
use crate::prove::preprocess::callback::ExternalCallback;
use crate::prove::types::Value;

/// An `ExternalCallback` that calls Python functions via PyO3.
pub struct PyExternalCallback<'py> {
    py: Python<'py>,
    callables: &'py Bound<'py, PyDict>,
}

impl<'py> PyExternalCallback<'py> {
    pub fn new(py: Python<'py>, callables: &'py Bound<'py, PyDict>) -> Self {
        Self { py, callables }
    }

    fn value_to_py(&self, val: &Value) -> PyObject {
        match val {
            Value::Integer(v) => v.into_pyobject(self.py).unwrap().unbind().into(),
            Value::Float(v) => v.into_pyobject(self.py).unwrap().unbind().into(),
            Value::Bool(v) => {
                (*v).into_pyobject(self.py).unwrap().to_owned().unbind().into()
            }
            Value::Str(s) => s.as_str().into_pyobject(self.py).unwrap().unbind().into(),
            Value::List(items) => {
                let py_list = pyo3::types::PyList::new(
                    self.py,
                    items.iter().map(|v| self.value_to_py(v)),
                )
                .unwrap();
                py_list.unbind().into()
            }
            Value::None => self.py.None(),
        }
    }

    fn py_to_value(&self, obj: &Bound<'py, pyo3::PyAny>) -> Result<Value, ProvingError> {
        // Order matters: bool before int (Python bool is a subclass of int)
        if let Ok(v) = obj.extract::<bool>() {
            return Ok(Value::Bool(v));
        }
        if let Ok(v) = obj.extract::<i64>() {
            return Ok(Value::Integer(v));
        }
        if let Ok(v) = obj.extract::<f64>() {
            return Ok(Value::Float(v));
        }
        if let Ok(v) = obj.extract::<String>() {
            return Ok(Value::Str(v));
        }
        if let Ok(v) = obj.extract::<i128>() {
            return Ok(Value::Integer(v as i64));
        }
        if obj.is_none() {
            return Ok(Value::None);
        }
        // Try list
        if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
            let items: Result<Vec<Value>, _> = list
                .iter()
                .map(|item| self.py_to_value(&item))
                .collect();
            return Ok(Value::List(items?));
        }
        Err(ProvingError::other(format!(
            "Cannot convert Python object to Value: {:?}", obj
        )))
    }
}

impl<'py> ExternalCallback for PyExternalCallback<'py> {
    fn call(&self, func_name: &str, args: Vec<Value>) -> Result<Value, ProvingError> {
        let callable = self.callables
            .get_item(func_name)
            .map_err(|e| ProvingError::other(format!("Failed to get '{}': {}", func_name, e)))?
            .ok_or_else(|| ProvingError::other(format!("External function '{}' not found", func_name)))?;

        let py_args: Vec<PyObject> = args.iter().map(|v| self.value_to_py(v)).collect();
        let py_tuple = PyTuple::new(self.py, &py_args)
            .map_err(|e| ProvingError::other(format!("Failed to create args tuple: {}", e)))?;

        let result = callable.call1(&py_tuple)
            .map_err(|e| ProvingError::other(format!("External function '{}' failed: {}", func_name, e)))?;

        self.py_to_value(&result)
    }
}
