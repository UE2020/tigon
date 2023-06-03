use shakmaty::*;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct FideHash(Color, Board);

pub fn fide_hash<B: Position>(pos: B) -> FideHash {
	FideHash(pos.turn(), pos.board().clone())
}