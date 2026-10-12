use noir_fpc::{divmod_by_pow2, qadd};

fn main(data: [Field; 8], bin_size: Field) -> pub [Field; 2] {
    let v0: Field = data[0];
    let v1: Field = 21888242871839275222246405745257275088548364400416034343688980814538953719809;
    let v2: Field = v0 - v1;
    v2.assert_max_bit_size::<64>();
    let v5: Field = data[1];
    let v6: Field = v5 - v1;
    v6.assert_max_bit_size::<64>();
    let v7: Field = data[2];
    let v8: Field = v7 - v1;
    v8.assert_max_bit_size::<64>();
    let v9: Field = data[3];
    let v10: Field = v9 - v1;
    v10.assert_max_bit_size::<64>();
    let v11: Field = data[4];
    let v12: Field = v11 - v1;
    v12.assert_max_bit_size::<64>();
    let v13: Field = data[5];
    let v14: Field = v13 - v1;
    v14.assert_max_bit_size::<64>();
    let v15: Field = data[6];
    let v16: Field = v15 - v1;
    v16.assert_max_bit_size::<64>();
    let v17: Field = data[7];
    let v18: Field = v17 - v1;
    v18.assert_max_bit_size::<64>();
    let v19: Field = bin_size;
    let v20: Field = v19 - v1;
    v20.assert_max_bit_size::<64>();
    let v21: Field = 4;
    assert(v19 == v21);
    let v22: Field = qadd(v17, v15);
    let v23: Field = qadd(v22, v13);
    let v24: Field = qadd(v23, v11);
    let v25: Field = qadd(v9, v7);
    let v26: Field = qadd(v25, v5);
    let v27: Field = qadd(v26, v0);
    let (v29, v30) = divmod_by_pow2::<2>(v24);
    let (v31, v32) = divmod_by_pow2::<2>(v27);
    [v29, v31]
}
