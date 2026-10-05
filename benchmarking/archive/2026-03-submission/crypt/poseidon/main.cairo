use core::hash::{HashStateExTrait, HashStateTrait};
use core::poseidon::{PoseidonTrait, poseidon_hash_span};

#[executable]
pub fn main(){
    let array_to_hash = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut hash = PoseidonTrait::new();
    hash = hash.update(*array_to_hash.at(0));
    hash = hash.update(*array_to_hash.at(1));
    hash = hash.update(*array_to_hash.at(2));
    hash = hash.update(*array_to_hash.at(3));
    hash = hash.update(*array_to_hash.at(4));
    hash = hash.update(*array_to_hash.at(5));
    hash = hash.update(*array_to_hash.at(6));
    hash = hash.update(*array_to_hash.at(7));
    hash = hash.update(*array_to_hash.at(8));
    hash = hash.update(*array_to_hash.at(9));
}