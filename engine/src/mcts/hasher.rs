use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use shakmaty::*;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct FideHash(Color, Board);

pub fn fide_hash<B: Position>(pos: B) -> FideHash {
    FideHash(pos.turn(), pos.board().clone())
}

pub fn default_hash<B: Position + Hash>(pos: &B) -> u64 {
    let mut hasher = DefaultHasher::new();
    pos.hash(&mut hasher);
    hasher.finish()
}
