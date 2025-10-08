// ElGamal encryption over BN254 (mod p arithmetic is implicit in Field type)

fn pow_field(base_in: Field, exp: Field) -> Field {
    let mut res: Field = 1;
    for i in 0..252 {
        res = res * res;
        let mask: u128 = (((exp as u128) >> ((251 - (i as u8)) as u8))) & 1;
        if mask == 1 {
            res = res * base_in;
        }
    }
    res
}

fn elgamal_keygen(g: Field, sk: Field) -> Field {
    // pk = g^sk
    pow_field(g, sk)
}

fn elgamal_encrypt(g: Field, pk: Field, msg: Field, r: Field) -> (Field, Field) {
    // c1 = g^r
    // c2 = msg * pk^r
    let c1 = pow_field(g, r);
    let c2 = msg * pow_field(pk, r);
    (c1, c2)
}

fn elgamal_decrypt(sk: Field, c1: Field, c2: Field) -> Field {
    // shared = c1^sk
    // msg = c2 * inv(shared)
    let shared = pow_field(c1, sk);
    let inv_shared = 1 / shared;
    let msg = c2 * inv_shared;
    msg
}

fn main(g: Field, sk: Field, r: Field, msg: Field) {
    // Key generation
    let pk = elgamal_keygen(g, sk);

    // Encryption
    let (c1, c2) = elgamal_encrypt(g, pk, msg, r);

    // Decryption
    let recovered = elgamal_decrypt(sk, c1, c2);

    // Round-trip check
    assert(recovered == msg, "Decryption failed: recovered != msg");
}
