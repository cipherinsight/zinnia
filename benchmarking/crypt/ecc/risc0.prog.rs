use risc0_zkvm::guest::env;
use ethereum_types::U512;

fn extended_gcd(a: U512, b: U512, p: U512) -> (U512, U512, U512) {
    if a == U512::from(0) {
        (b, U512::from(0), U512::from(1))
    } else {
        let (g, y, x) = extended_gcd(b % a, a, p);
        // println!("Backtrack! g={:?}, y={:?}, x={:?}, a={:?}, b={:?}", g, y, x, a, b);
        // println!("step 1!! {:?}", (b / a) * y);
        (g, (x + (p - ((((b / a) % p) * y) % p)) % p) % p, y)
    }
}

fn modular_inverse(a: U512, p: U512) -> U512 {
    let (gcd, x, _) = extended_gcd(a % p, p, p);
    assert!(gcd == U512::from(1));
    return (x % p + p) % p;
}

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let x1 = U512([env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(),]);
    let y1 = U512([env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(),]);
    let x2 = U512([env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(),]);
    let y2 = U512([env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(),]);
    let x3 = U512([env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(),]);
    let y3 = U512([env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(), env::read(),]);

    let P = U512::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    let a = U512::from(168700);
    let neg_a: U512 = P - a;
    let d = U512::from(168696);
    let one = U512::from(1);

    // point 1 should on the curve
    let x1_square = (x1 * x1) % P;
    let y1_square = (y1 * y1) % P;
    let left= (((a * x1_square) % P) + y1_square) % P;
    let right = one + (d * ((x1_square * y1_square) % P)) % P;
    assert_eq!(left % P, right % P);

    // point 2 should on the curve
    let x2_square = (x2 * x2) % P;
    let y2_square = (y2 * y2) % P;
    let left= (((a * x2_square) % P) + y2_square) % P;
    let right = one + (d * ((x2_square * y2_square) % P)) % P;
    assert_eq!(left % P, right % P);

    // point 3 should on the curve
    let x3_square = (x3 * x3) % P;
    let y3_square = (y3 * y3) % P;
    let left= (((a * x3_square) % P) + y3_square) % P;
    let right = one + (d * ((x3_square * y3_square) % P)) % P;
    assert_eq!(left % P, right % P);

    // add p1 and p2 together
    let beta = (x1 * y2) % P;
    let gamma = (y1 * x2) % P;
    let delta = (((((neg_a * x1) % P) + y1) % P) * ((x2 + y2) % P)) % P;
    let tau = (beta * gamma) % P;
    let tmp = (one + ((d * tau) % P)) % P;
    let x4 = ((beta + gamma) % P) * modular_inverse(tmp, P);
    let y4 = ((((delta + ((a * beta) % P)) % P) + (P - gamma) % P) % P) * modular_inverse((one + (P - ((d * tau) % P)) % P) % P, P);

    // verify equality
    assert_eq!(x4 % P, x3);
    assert_eq!(y4 % P, y3);

    // write public output to the journal
    // env::commit(&input);
}
