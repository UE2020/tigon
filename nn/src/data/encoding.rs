use shakmaty::*;

type EncodedPositions = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>;

fn coords(mut sq: Square, flip: bool) -> (usize, usize) {
    if flip {
        sq = Square::new((sq as u8 ^ 0x38) as u32);
    }

    (sq.rank() as usize, sq.file() as usize)
}

fn icoords(sq: Square) -> (isize, isize) {
    (sq.rank() as isize, sq.file() as isize)
}

pub fn encode_positions<B: Position>(pos: &B) -> EncodedPositions {
    let mut planes = ndarray::Array::<f32, _>::zeros((22, 8, 8));
    let flip = pos.turn() == Color::Black;
    let pawns = pos.board().pawns();
    let knights = pos.board().knights();
    let bishops = pos.board().bishops();
    let rooks = pos.board().rooks();
    let queens = pos.board().queens();
    let kings = pos.board().kings();

    let mut white = pos.board().white();
    let mut black = pos.board().black();

    if flip {
        std::mem::swap(&mut white, &mut black);
    }

    //////////////////// pawns ////////////////////

    let mut remaining = white & pawns;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[0, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[1, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[6, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[7, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[4, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[5, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[2, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[3, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[8, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[9, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    //////////////////// kings ////////////////////

    let mut remaining = white & kings;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[10, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut remaining = black & kings;
    while remaining != Bitboard(0) {
        let sq = Square::new(remaining.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[11, r, f]] = 1.0;

        remaining ^= Bitboard::from_square(sq);
    }

    let mut white_color = Color::White;
    let mut black_color = Color::Black;

    if flip {
        std::mem::swap(&mut white_color, &mut black_color);
    }

    if pos.castles().has(white_color, CastlingSide::KingSide) {
        for x in 0..8 {
            for y in 0..8 {
                planes[[12, x, y]] = 1.0;
            }
        }
    }

    if pos.castles().has(black_color, CastlingSide::KingSide) {
        for x in 0..8 {
            for y in 0..8 {
                planes[[13, x, y]] = 1.0;
            }
        }
    }

    if pos.castles().has(white_color, CastlingSide::QueenSide) {
        for x in 0..8 {
            for y in 0..8 {
                planes[[14, x, y]] = 1.0;
            }
        }
    }

    if pos.castles().has(black_color, CastlingSide::QueenSide) {
        for x in 0..8 {
            for y in 0..8 {
                planes[[15, x, y]] = 1.0;
            }
        }
    }

	let pawn_difference = (pawns & white).count() - (pawns & black).count() + 8;
	let knight_difference = (knights & white).count() - (knights & black).count() + 10;
	let bishop_difference = (bishops & white).count() - (bishops & black).count() + 10;
	let rook_difference = (rooks & white).count() - (rooks & black).count() + 10;
	let queen_difference = (queens & white).count() - (queens & black).count() + 9;

	for x in 0..8 {
		for y in 0..8 {
			planes[[16, x, y]] = (pawn_difference as f32) / 16.0;
		}
	}

	for x in 0..8 {
		for y in 0..8 {
			planes[[17, x, y]] = (knight_difference as f32) / 20.0;
		}
	}

	for x in 0..8 {
		for y in 0..8 {
			planes[[18, x, y]] = (bishop_difference as f32) / 20.0;
		}
	}

	for x in 0..8 {
		for y in 0..8 {
			planes[[19, x, y]] = (rook_difference as f32) / 20.0;
		}
	}

	for x in 0..8 {
		for y in 0..8 {
			planes[[20, x, y]] = (queen_difference as f32) / 18.0;
		}
	}

	let king_sq = Square::new((white | kings).0.trailing_zeros());
	let mut checkers = pos.king_attackers(king_sq, black_color, black | white);
    while checkers != Bitboard(0) {
        let sq = Square::new(checkers.0.trailing_zeros());
        let (r, f) = coords(sq, flip);
        planes[[21, r, f]] = 1.0;

        checkers ^= Bitboard::from_square(sq);
    }

    planes
}

#[allow(unused_assignments)]
pub fn move_to_idx(mov: &Move, flip: bool) -> (isize, isize, isize) {
    let (from, to) = match mov {
        Move::Castle { king, rook } => (
            *king,
            match mov.castling_side().unwrap() {
                CastlingSide::KingSide => rook.offset(-1).unwrap(),
                CastlingSide::QueenSide => rook.offset(1).unwrap(),
            },
        ),
        m => (m.from().unwrap(), m.to()),
    };

    let (from_rank, from_file) = icoords(if flip {
        Square::new((from as u8 ^ 0x38) as u32)
    } else {
        from
    });

    let (to_rank, to_file) = icoords(if flip {
        Square::new((to as u8 ^ 0x38) as u32)
    } else {
        to
    });

    let mut direction_plane = 0;
    let mut distance = 0;
    let mut direction_and_distance_plane = 0;

    if from_rank == to_rank && from_file < to_file {
        direction_plane = 0;
        distance = to_file - from_file;
        direction_and_distance_plane = direction_plane + distance
    } else if from_rank == to_rank && from_file > to_file {
        direction_plane = 8;
        distance = from_file - to_file;
        direction_and_distance_plane = direction_plane + distance
    } else if from_file == to_file && from_rank < to_rank {
        direction_plane = 16;
        distance = to_rank - from_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if from_file == to_file && from_rank > to_rank {
        direction_plane = 24;
        distance = from_rank - to_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == to_rank - from_rank && to_file - from_file > 0 {
        direction_plane = 32;
        distance = to_rank - from_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == to_rank - from_rank && to_file - from_file < 0 {
        direction_plane = 40;
        distance = from_rank - to_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == -(to_rank - from_rank) && to_file - from_file > 0 {
        direction_plane = 48;
        distance = to_file - from_file;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == -(to_rank - from_rank) && to_file - from_file < 0 {
        direction_plane = 56;
        distance = from_file - to_file;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == 1 && to_rank - from_rank == 2 {
        direction_and_distance_plane = 64
    } else if to_file - from_file == 2 && to_rank - from_rank == 1 {
        direction_and_distance_plane = 65
    } else if to_file - from_file == 2 && to_rank - from_rank == -1 {
        direction_and_distance_plane = 66
    } else if to_file - from_file == 1 && to_rank - from_rank == -2 {
        direction_and_distance_plane = 67
    } else if to_file - from_file == -1 && to_rank - from_rank == 2 {
        direction_and_distance_plane = 68
    } else if to_file - from_file == -2 && to_rank - from_rank == 1 {
        direction_and_distance_plane = 69
    } else if to_file - from_file == -2 && to_rank - from_rank == -1 {
        direction_and_distance_plane = 70
    } else if to_file - from_file == -1 && to_rank - from_rank == -2 {
        direction_and_distance_plane = 71
    }

    (direction_and_distance_plane, from_rank, from_file)
}

type MoveMasks = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>;

pub fn legal_move_masks<B: Position>(pos: &B) -> MoveMasks {
    let mut mask = ndarray::Array::<f32, _>::zeros((72, 8, 8));

    let flip = pos.turn() == Color::Black;

    let moves = pos.legal_moves();
    for mov in moves {
        let (plane_idx, rank_idx, file_idx) = move_to_idx(&mov, flip);
        mask[[plane_idx as usize, rank_idx as usize, file_idx as usize]] = 1.0;
    }

    mask
}
