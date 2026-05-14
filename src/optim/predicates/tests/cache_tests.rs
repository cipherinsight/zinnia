//! Tests for the discharge result cache (`cache.rs`).

use crate::optim::predicates::{DischargeCache, DischargeKey, DischargeResult};

#[test]
fn new_cache_is_empty() {
    let c = DischargeCache::new();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
}

#[test]
fn insert_then_get_roundtrips() {
    let mut c = DischargeCache::new();
    let key = DischargeKey::new(7, 0xdeadbeef, 0x12345678);
    assert!(c.get(&key).is_none());
    c.insert(key.clone(), DischargeResult::Proved);
    assert_eq!(c.get(&key), Some(DischargeResult::Proved));
    assert_eq!(c.len(), 1);
}

#[test]
fn distinct_keys_do_not_collide() {
    let mut c = DischargeCache::new();
    let k1 = DischargeKey::new(1, 100, 200);
    let k2 = DischargeKey::new(1, 100, 201);
    let k3 = DischargeKey::new(2, 100, 200);
    c.insert(k1.clone(), DischargeResult::Proved);
    c.insert(k2.clone(), DischargeResult::Disproved);
    c.insert(k3.clone(), DischargeResult::Unknown);
    assert_eq!(c.get(&k1), Some(DischargeResult::Proved));
    assert_eq!(c.get(&k2), Some(DischargeResult::Disproved));
    assert_eq!(c.get(&k3), Some(DischargeResult::Unknown));
    assert_eq!(c.len(), 3);
}

#[test]
fn insert_overwrites_existing_key() {
    let mut c = DischargeCache::new();
    let key = DischargeKey::new(0, 0, 0);
    c.insert(key.clone(), DischargeResult::Unknown);
    c.insert(key.clone(), DischargeResult::Proved);
    assert_eq!(c.get(&key), Some(DischargeResult::Proved));
    assert_eq!(c.len(), 1);
}

#[test]
fn clear_empties_cache() {
    let mut c = DischargeCache::new();
    c.insert(DischargeKey::new(0, 0, 0), DischargeResult::Proved);
    c.clear();
    assert!(c.is_empty());
}
