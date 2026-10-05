//! NumPy matmul for static batch dimensions, including vector promotion and
//! empty axes. Work directly with array metadata so empty shapes are retained.
use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};
use crate::helpers::shape_arith::{broadcast_shapes, decode_coords, row_major_strides};

pub fn matmul(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let ls = crate::helpers::composite::get_composite_shape(lhs);
    let rs = crate::helpers::composite::get_composite_shape(rhs);
    assert!(!ls.is_empty() && !rs.is_empty(), "matmul requires arrays, not scalars");
    let lv = ls.len() == 1;
    let rv = rs.len() == 1;
    let k = *ls.last().unwrap();
    let rk = if rv { rs[0] } else { rs[rs.len()-2] };
    assert_eq!(k, rk, "matmul contraction dimensions must match");
    let m = if lv { 1 } else { ls[ls.len()-2] };
    let n = if rv { 1 } else { rs[rs.len()-1] };
    let lb = if lv { &[][..] } else { &ls[..ls.len()-2] };
    let rb = if rv { &[][..] } else { &rs[..rs.len()-2] };
    let batch = broadcast_shapes(lb,rb).expect("matmul batch dimensions must broadcast");
    let mut shape = batch.clone();
    if !lv { shape.push(m); }
    if !rv { shape.push(n); }
    let cells = |b: &mut IRBuilder, v: &Value| {
        if matches!(v, Value::StaticArray { .. }) { super::elementwise::payload_cells(b,v) }
        else { crate::helpers::composite::flatten_composite(v) }
    };
    let a = cells(b,lhs); let other = cells(b,rhs);
    let float = matches!(lhs,Value::StaticArray {dtype:NumberType::Float,..})
        || matches!(rhs,Value::StaticArray {dtype:NumberType::Float,..})
        || a.iter().chain(other.iter()).any(|v| matches!(v,Value::Float(_)));
    let complex = matches!(lhs,Value::StaticArray {dtype:NumberType::Complex,..})
        || matches!(rhs,Value::StaticArray {dtype:NumberType::Complex,..})
        || a.iter().chain(other.iter()).any(|v| matches!(v,Value::Complex {..}));
    let strides = row_major_strides(&batch);
    let offset = |coords: &[usize], source: &[usize]| -> usize {
        let start = coords.len()-source.len();
        source.iter().enumerate().fold(0,|flat,(axis,&dim)| flat*dim+if dim==1 {0} else {coords[start+axis]})
    };
    let mut out = Vec::new();
    for bi in 0..batch.iter().product::<usize>() {
        let coords = decode_coords(bi,&batch,&strides);
        let ab = offset(&coords,lb)*m*k;
        let bb = offset(&coords,rb)*k*n;
        for i in 0..m {
            for j in 0..n {
                let row: Vec<Value> = (0..k).map(|q| a[ab+i*k+q].clone()).collect();
                let col: Vec<Value> = (0..k).map(|q| other[bb+q*n+j].clone()).collect();
                out.push(if complex {
                    crate::ops::static_ndarray_ops::matmul_dot_complex(b,&row,&col)
                } else { crate::ops::static_ndarray_ops::matmul_dot(b,&row,&col,float) });
            }
        }
    }
    if shape.is_empty() { return out.remove(0); }
    if complex {
        let (re,im) = out.into_iter().map(|v| match v {
            Value::Complex {real,imag} => (Value::Float(real),Value::Float(imag)),
            _ => unreachable!(),
        }).unzip();
        super::base::build_static_array_from_flat_complex(b,re,im,shape)
    } else {
        super::base::build_static_array_from_flat(b,out,shape,if float {NumberType::Float} else {NumberType::Integer})
    }
}
