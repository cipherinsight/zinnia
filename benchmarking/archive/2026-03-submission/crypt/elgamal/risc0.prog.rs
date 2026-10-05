#![no_main]
use risc0_zkvm::guest::env;
use ethereum_types::U512;

// BN254 field modulus
const P_STR: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

fn add_mod(a: &U512, b: &U512, p: &U512) -> U512 {
    let mut res = (a + b) % *p;
    if res >= *p { res -= *p; }
    res
}

fn mul_mod(a: &U512, b: &U512, p: &U512) -> U512 {
    (a * b) % *p
}

fn mod_pow(mut base: U512, mut exp: U512, p: &U512) -> U512 {
    let mut result = U512::from(1u64);
    base = base % *p;
    while exp > U512::from(0u64) {
        if (exp & U512::from(1u64)) == U512::from(1u64) {
            result = (result * &base) % *p;
        }
        base = (&base * &base) % *p;
        exp >>= 1;
    }
    result
}

fn extended_gcd(a: U512, b: U512, p: &U512) -> (U512, U512, U512) {
    if a == U512::from(0u64) {
        (b, U512::from(0u64), U512::from(1u64))
    } else {
        let (g, y, x) = extended_gcd(b % a, a, p);
        (g, (x + (p - ((((b / a) % p) * y) % p)) % p) % p, y)
    }
}

fn mod_inv(a: U512, p: &U512) -> U512 {
    let (gcd, x, _) = extended_gcd(a % *p, *p, p);
    assert!(gcd == U512::from(1u64));
    (x % *p + *p) % *p
}

fn elgamal_keygen(g: &U512, sk: &U512, p: &U512) -> U512 {
    mod_pow(g.clone(), sk.clone(), p)
}

fn elgamal_encrypt(g: &U512, pk: &U512, msg: &U512, r: &U512, p: &U512) -> (U512, U512) {
    let c1 = mod_pow(g.clone(), r.clone(), p);
    let c2 = mul_mod(msg, &mod_pow(pk.clone(), r.clone(), p), p);
    (c1, c2)
}

fn elgamal_decrypt(sk: &U512, c1: &U512, c2: &U512, p: &U512) -> U512 {
    let shared = mod_pow(c1.clone(), sk.clone(), p);
    let inv_shared = mod_inv(shared, p);
    mul_mod(c2, &inv_shared, p)
}

risc0_zkvm::guest::entry!(main);
pub fn main() {
    let g = env::read::<U512>();
    let sk = env::read::<U512>();
    let r = env::read::<U512>();
    let msg = env::read::<U512>();

    let p = U512::from_str_radix(P_STR, 10).unwrap();

    let pk = elgamal_keygen(&g, &sk, &p);
    let (c1, c2) = elgamal_encrypt(&g, &pk, &msg, &r, &p);
    let recovered = elgamal_decrypt(&sk, &c1, &c2, &p);

    assert_eq!(recovered % p, msg % p);

    // env::commit(&recovered);
}
