// npbench_deep_learning_resnet_resnet__resnet_basicblock.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Convolution materializes rounded elementwise products, then sums in axis order.
// std: mean, centered rounded squares, mean, sqrt; denominator is sqrt(std+eps).
// Q32 throughout: float32 array storage adds no rounding step. All indices are static.
// @outputs: float[N*H3*W3*CO]
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 4; // = input_0
global H: u32 = 7; // = input_1
global W: u32 = 7; // = input_2
global CI: u32 = 16; // = input_3
global CM: u32 = 4; // = conv1_3
global K1: u32 = 1; // = conv1_0
global K2: u32 = 3; // = conv2_0
global K3: u32 = 1; // = conv3_0
global CO: u32 = 16; // = conv3_3
global PH: u32 = H + 2;
global PW: u32 = W + 2;
global H1: u32 = H - K1 + 1;
global W1: u32 = W - K1 + 1;
global H2: u32 = PH - K2 + 1;
global W2: u32 = PW - K2 + 1;
global H3: u32 = H2 - K3 + 1;
global W3: u32 = W2 - K3 + 1;

fn main(input: [Field; N*H*W*CI], conv1: [Field; K1*K1*CI*CM],
        conv2: [Field; K2*K2*CM*CM], conv3: [Field; K3*K3*CM*CO]) -> pub [Field; N*H3*W3*CO] {
    noir_hw::check_inputs(input);
    noir_hw::check_inputs(conv1);
    noir_hw::check_inputs(conv2);
    noir_hw::check_inputs(conv3);
    let mut padded = [0; N*PH*PW*CM];
    let first = conv2d::<H,W,CI,CM,K1,H1,W1>(input,conv1);
    for b in 0..N { for i in 0..H1 { for j in 0..W1 { for c in 0..CM {
        padded[((b*PH+i+1)*PW+j+1)*CM+c] = first[((b*H1+i)*W1+j)*CM+c];
    }}}}
    let x = batchnorm2d(padded);
    let x = relu(x);
    let x = conv2d::<PH,PW,CM,CM,K2,H2,W2>(x,conv2);
    let x = batchnorm2d(x);
    let x = relu(x);
    let x = conv2d::<H2,W2,CM,CO,K3,H3,W3>(x,conv3);
    let x = batchnorm2d(x);
    let mut sum = [0; N*H3*W3*CO];
    for i in 0..N*H3*W3*CO { sum[i] = x[i]+input[i]; }
    relu(sum)
}
fn relu<let S: u32>(x: [Field; S]) -> [Field; S] {
    let mut result = [0; S];
    for i in 0..S { result[i] = noir_hw::max(x[i],0); }
    result
}
fn mean<let S: u32>(x: [Field; S]) -> [Field; S/N] {
    let mut result = [0; S/N];
    for i in 0..S/N {
        let mut sum: Field = 0;
        for b in 0..N { sum += x[b*(S/N)+i]; }
        result[i] = noir_hw::divmod_pos(sum,N as Field).0;
    }
    result
}
fn batchnorm2d<let S: u32>(x: [Field; S]) -> [Field; S] {
    let avg = mean(x);
    let std_avg = mean(x);
    let mut deviations = [0; S];
    for i in 0..S { deviations[i] = x[i]-std_avg[i%(S/N)]; }
    let mut squares = [0; S];
    for i in 0..S { squares[i] = noir_hw::mul(deviations[i],deviations[i]); }
    let variance = mean(squares);
    let mut std = [0; S/N];
    for i in 0..S/N { std[i] = noir_hw::sqrt(variance[i]); }
    let mut numerator = [0; S];
    for i in 0..S { numerator[i] = x[i]-avg[i%(S/N)]; }
    let mut shifted = [0; S/N];
    for i in 0..S/N { shifted[i] = std[i]+42950; } // Q32(1e-5)
    let mut denominator = [0; S/N];
    for i in 0..S/N { denominator[i] = noir_hw::sqrt(shifted[i]); }
    let mut result = [0; S];
    for i in 0..S { result[i] = noir_hw::div(numerator[i],denominator[i%(S/N)]); }
    result
}
fn conv2d<let HH: u32, let WW: u32, let IC: u32, let OC: u32, let K: u32, let OH: u32, let OW: u32>(
    input: [Field; N*HH*WW*IC], weights: [Field; K*K*IC*OC]) -> [Field; N*OH*OW*OC] {
    let mut output = [0; N*OH*OW*OC];
    for i in 0..OH { for j in 0..OW {
        let mut products = [0; N*K*K*IC*OC];
        for b in 0..N { for a in 0..K { for d in 0..K { for c in 0..IC { for o in 0..OC {
            products[(((b*K+a)*K+d)*IC+c)*OC+o] = noir_hw::mul(
                input[((b*HH+i+a)*WW+j+d)*IC+c], weights[((a*K+d)*IC+c)*OC+o]);
        }}}}}
        for b in 0..N { for o in 0..OC {
            let mut sum: Field = 0;
            for a in 0..K { for d in 0..K { for c in 0..IC {
                sum += products[(((b*K+a)*K+d)*IC+c)*OC+o];
            }}}
            output[((b*OH+i)*OW+j)*OC+o] = sum;
        }}
    }}
    output
}
