// pythran_ccopy__CCOPY: source unit-stride and general-stride loops, in order.
// Keep all 1024 steps (zero strides can repeat a valid copy 1024 times).
// Guard inactive accesses with index zero. Shared Q32 inputs are copied exactly.
// @outputs: float[Y_CELLS]
global NX: u32 = 4; // = CX_0
global NY: u32 = 4; // = CY_0
global PARTS: u32 = 2; // complex real/imaginary components, not a shape dimension
global X_CELLS: u32 = PARTS * NX;
global Y_CELLS: u32 = PARTS * NY;
global Y_INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (CY_0 - 1) // 128)
global MAX_STEPS: u32 = 1024; // source precondition
fn main(N: Field, CX: [Field; X_CELLS], INCX: Field, CY: [Field; Y_CELLS], INCY: Field) -> pub [Field; Y_CELLS] {
    let nw = noir_hw::window(N);
    let xw = noir_hw::window(INCX);
    let yw = noir_hw::window(INCY);
    noir_hw::check_inputs(CX);
    noir_hw::check_inputs(CY);
    assert(nw >= noir_hw::window(0));
    assert(nw <= noir_hw::window(MAX_STEPS as Field));
    let unit = (INCX == 1) & (INCY == 1);
    let mut cy = CY;
    for step in 0..MAX_STEPS {
        let active = unit & (nw > noir_hw::window(step as Field));
        // Constant-fold unit-stride indices; out-of-range active steps must fail.
        if (step < NX) & (step < NY) {
            for part in 0..PARTS {
                cy[PARTS * step + part] = if active { CX[PARTS * step + part] } else { cy[PARTS * step + part] };
            }
        } else { assert(!active); }
    }
    let one_minus_n = 1 - N;
    let mut ix = if xw < noir_hw::window(0) { one_minus_n * INCX } else { 0 };
    let mut iy = if yw < noir_hw::window(0) { one_minus_n * INCY } else { 0 };
    let mut components = [[0; NX]; PARTS];
    for part in 0..PARTS {
        for i in 0..NX { components[part][i] = CX[PARTS * i + part]; }
    }
    for step in 0..MAX_STEPS {
        let active = !unit & (nw > noir_hw::window(step as Field));
        let safe_x = if active { ix } else { 0 };
        let safe_y = if active { iy } else { 0 };
        let destination = noir_hw::index_pos::<Y_INDEX_BITS>(safe_y, NY);
        let mut value = [0; PARTS];
        for part in 0..PARTS { value[part] = noir_hw::pick(components[part], safe_x); }
        // A short destination is cheaper as conditional cell writes than a RAM update.
        for j in 0..NY {
            let write = active & (destination == j);
            for part in 0..PARTS {
                cy[PARTS * j + part] = if write { value[part] } else { cy[PARTS * j + part] };
            }
        }
        ix = if active { ix + INCX } else { ix };
        iy = if active { iy + INCY } else { iy };
    }
    cy
}
