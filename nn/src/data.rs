use dfdx::data::*;
use pgn_reader::*;
use shakmaty::*;

mod encoding;
pub use encoding::*;

#[derive(Debug, Clone)]
pub struct PgnVisitor {
    positions: Vec<(Chess, Option<Move>)>,
    outcome: Option<Outcome>,
    termination_normal: bool,
}

impl PgnVisitor {
    pub fn new() -> Self {
        Self {
            positions: vec![],
            outcome: None,
            termination_normal: true,
        }
    }
}

impl Visitor for PgnVisitor {
    type Result = Option<(Vec<(Chess, Move)>, Outcome)>;

    fn begin_game(&mut self) {
        self.positions = vec![(Chess::default(), None)];
        self.termination_normal = false;
    }

    fn san(&mut self, san_plus: SanPlus) {
        let last = self.positions.last_mut().unwrap();
        let mov = san_plus.san.to_move(&last.0).expect("invalid move");
        last.1 = Some(mov.clone());

        let new_pos = last.0.clone().play(&mov).expect("invalid move");
        self.positions.push((new_pos, None));
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {
        let mut positions = self.positions.clone();
        positions.pop();
        let positions = positions.into_iter().map(|(pos, mov)| (pos, mov.unwrap()));
        Some((positions.collect(), self.outcome.unwrap()))
    }

    fn outcome(&mut self, outcome: Option<Outcome>) {
        self.outcome = outcome;
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        if key == b"Termination" {
            if value.as_bytes() != b"Normal" {
                self.termination_normal = true;
            }
        }
    }
}

pub struct ChessPositionSet(Vec<((Chess, Move), Outcome)>);

impl ChessPositionSet {
    pub fn new(games: Vec<(Vec<(Chess, Move)>, Outcome)>) -> Self {
        let mut positions = Vec::new();
        for game in games {
            for (i, pos) in game.0.into_iter().enumerate() {
                positions.push((pos, game.1));
            }
        }

        Self(positions)
    }
}

impl ExactSizeDataset for ChessPositionSet {
    // board, value, policy, mask
    type Item<'a> = (Vec<f32>, f32, Vec<f32>, Vec<f32>) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        let ((pos, mov), outcome) = &self.0[index];
        let data = encode_positions(pos);

        // create policy output
        let mut output = vec![0.0; 4608];
        let (plane_idx, rank_idx, file_idx) = move_to_idx(&mov, pos.turn() == Color::Black);
        let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
        output[mov_idx as usize] = 1.0;

        (
            data.into_raw_vec(),
            match outcome {
                Outcome::Decisive { winner } => {
                    turn_to_side(*winner) as f32 * turn_to_side(pos.turn()) as f32
                }
                Outcome::Draw => 0.0,
            },
            output,
            legal_move_masks(pos).into_raw_vec(),
        )
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

pub fn turn_to_side(color: Color) -> i8 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}
