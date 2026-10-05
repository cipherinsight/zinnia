//! The kernel's return value: which cells become public instances and how to
//! decode them.
//!
//! Every output that depends on a private input is a public instance cell.
//! Outputs that are compile-time constants of the circuit (for example a shape
//! fixed by the kernel's signature) are not instances: the verifier knows them
//! from the circuit itself, exactly as Zinnia leaves constant outputs out of its
//! public instance. They are still reported in `publics` so the output check
//! sees the whole return value.
//!
//! A data-dependent length (for example `x[x >= 0]` or `np.argwhere`) is exposed
//! as one public length cell followed by the values padded with zeros to the
//! maximum length.

use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::q32::{dequantize, signed, Cell};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Float,
    Int,
    Bool,
}

impl Kind {
    fn decode(self, x: &Fr) -> Value {
        match self {
            Kind::Float => json!(dequantize(x)),
            Kind::Int => json!(signed(x) as i64),
            Kind::Bool => json!(signed(x) != 0),
        }
    }
}

/// One piece of the return value, without its cells: what the verifier side
/// needs to decode an instance vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "segment", rename_all = "lowercase")]
pub enum Segment {
    /// `count` public cells.
    Cells { count: usize, kind: Kind },
    /// Values fixed by the circuit; no public cells.
    Constants { values: Vec<Value>, kind: Kind },
    /// One public length cell (number of items), then `items * unit` public
    /// cells of which the first `length * unit` are the value.
    Dynamic { items: usize, unit: usize, kind: Kind },
    /// A 2-D output of data-dependent shape: two public cells `(rows, cols)`,
    /// then `max_rows * max_cols` public cells of which the first
    /// `rows * cols` are the value, row-major and compact.
    Dynamic2 { max_rows: usize, max_cols: usize, kind: Kind },
}

/// The decodable layout of the public instance.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Layout {
    pub segments: Vec<Segment>,
}

impl Layout {
    /// Decodes an instance vector into the kernel's flattened return value.
    pub fn decode(&self, instance: &[Fr]) -> Result<Vec<Value>, String> {
        let mut out = Vec::new();
        let mut rest = instance;
        let mut take = |n: usize| -> Result<&[Fr], String> {
            if rest.len() < n {
                return Err("instance shorter than the output layout".into());
            }
            let (head, tail) = rest.split_at(n);
            rest = tail;
            Ok(head)
        };
        for seg in &self.segments {
            match seg {
                Segment::Cells { count, kind } => out.extend(take(*count)?.iter().map(|x| kind.decode(x))),
                Segment::Constants { values, .. } => out.extend(values.iter().cloned()),
                Segment::Dynamic { items, unit, kind } => {
                    let len = signed(&take(1)?[0]);
                    if len < 0 || len as usize > *items {
                        return Err(format!("dynamic length {len} outside [0, {items}]"));
                    }
                    let values = take(items * unit)?;
                    out.extend(values[..len as usize * unit].iter().map(|x| kind.decode(x)));
                }
                Segment::Dynamic2 { max_rows, max_cols, kind } => {
                    let dims = take(2)?;
                    let (rows, cols) = (signed(&dims[0]), signed(&dims[1]));
                    if !(0..=*max_rows as i128).contains(&rows) || !(0..=*max_cols as i128).contains(&cols) {
                        return Err(format!("dynamic shape ({rows}, {cols}) outside ({max_rows}, {max_cols})"));
                    }
                    let values = take(max_rows * max_cols)?;
                    out.extend(values[..(rows * cols) as usize].iter().map(|x| kind.decode(x)));
                }
            }
        }
        if !rest.is_empty() {
            return Err("instance longer than the output layout".into());
        }
        Ok(out)
    }
}

/// Outputs in kernel return order (flattened row-major), collected by a circuit.
#[derive(Clone, Debug, Default)]
pub struct Outputs {
    layout: Layout,
    cells: Vec<Cell>,
}

impl Outputs {
    pub fn floats(&mut self, cells: &[Cell]) {
        self.push_cells(cells, Kind::Float);
    }

    pub fn float(&mut self, cell: Cell) {
        self.push_cells(&[cell], Kind::Float);
    }

    pub fn ints(&mut self, cells: &[Cell]) {
        self.push_cells(cells, Kind::Int);
    }

    pub fn int(&mut self, cell: Cell) {
        self.push_cells(&[cell], Kind::Int);
    }

    pub fn bools(&mut self, cells: &[Cell]) {
        self.push_cells(cells, Kind::Bool);
    }

    pub fn bool(&mut self, cell: Cell) {
        self.push_cells(&[cell], Kind::Bool);
    }

    /// Integer outputs fixed by the circuit (not public instances).
    pub fn const_ints(&mut self, values: &[i64]) {
        let values = values.iter().map(|v| json!(v)).collect();
        self.layout.segments.push(Segment::Constants { values, kind: Kind::Int });
    }

    /// Float outputs fixed by the circuit (not public instances).
    pub fn const_floats(&mut self, values: &[f64]) {
        let values = values.iter().map(|v| json!(v)).collect();
        self.layout.segments.push(Segment::Constants { values, kind: Kind::Float });
    }

    /// A data-dependent-length output: `len` items of `unit` cells each at the
    /// front of `cells`, the rest zero padding.
    pub fn dynamic(&mut self, len: Cell, cells: &[Cell], kind: Kind, unit: usize) {
        assert!(unit > 0 && cells.len() % unit == 0);
        self.layout.segments.push(Segment::Dynamic { items: cells.len() / unit, unit, kind });
        self.cells.push(len);
        self.cells.extend_from_slice(cells);
    }

    /// A 2-D output of data-dependent shape `(rows, cols)` stored compact,
    /// row-major, at the front of `cells` (`max_rows * max_cols` cells).
    pub fn dynamic2(&mut self, rows: Cell, cols: Cell, cells: &[Cell], kind: Kind, max_rows: usize, max_cols: usize) {
        assert_eq!(cells.len(), max_rows * max_cols);
        self.layout.segments.push(Segment::Dynamic2 { max_rows, max_cols, kind });
        self.cells.extend_from_slice(&[rows, cols]);
        self.cells.extend_from_slice(cells);
    }

    fn push_cells(&mut self, cells: &[Cell], kind: Kind) {
        self.layout.segments.push(Segment::Cells { count: cells.len(), kind });
        self.cells.extend_from_slice(cells);
    }

    /// The public instance column, in order.
    pub fn instance_cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}
