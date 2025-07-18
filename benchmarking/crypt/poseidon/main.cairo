use core::hash::{HashStateExTrait, HashStateTrait};
use core::poseidon::{PoseidonTrait, poseidon_hash_span};

#[executable]
pub fn main(){
    let array_to_hash = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut hash = PoseidonTrait::new();
    for i in 0..10_u32 {
        hash = hash.update(*array_to_hash.at(i));
    }
}