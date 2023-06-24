use dfdx::data::*;
use pgn_reader::*;
use regex::Regex;
use shakmaty::*;

mod encoding;
pub use encoding::*;

#[derive(Debug, Clone)]
pub struct PgnVisitor {
    positions: Vec<(Chess, Option<Move>, Option<f32>)>,
    //outcome: Option<Outcome>,
    re: Regex,
    termination_normal: bool,
    total_elo: u16,
}

impl PgnVisitor {
    pub fn new() -> Self {
        Self {
            positions: vec![],
            //outcome: None,
            termination_normal: true,
            re: Regex::new(r"([-+]|#)?[0-9]*\.?[0-9]+").unwrap(),
            total_elo: 0,
        }
    }
}

impl Visitor for PgnVisitor {
    type Result = Option<(u16, Vec<(Chess, Move, Option<f32>)>)>;

    fn begin_game(&mut self) {
        self.positions = vec![(Chess::default(), None, None)];
        self.termination_normal = false;
        self.total_elo = 0;
    }

    fn san(&mut self, san_plus: SanPlus) {
        let last = self.positions.last_mut().unwrap();
        let mov = san_plus.san.to_move(&last.0).expect("invalid move");
        last.1 = Some(mov.clone());

        let new_pos = last.0.clone().play(&mov).expect("invalid move");
        self.positions.push((new_pos, None, None));
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {
        let mut positions = self.positions.clone();
        positions.pop();
        let positions = positions
            .into_iter()
            .map(|(pos, mov, eval)| (pos, mov.unwrap(), eval));
        Some((self.total_elo / 2, positions.collect()))
    }

    fn outcome(&mut self, outcome: Option<Outcome>) {
        //self.outcome = outcome;
    }

    fn comment(&mut self, comment: RawComment<'_>) {
        let comment = comment.as_bytes();
        let comment = std::str::from_utf8(comment).expect("invalid utf8");
        let captures = self.re.captures(comment).unwrap();
        let matched_capture = captures.get(0).expect("no captures in comment");
        let eval: f32 = {
            let text = matched_capture.as_str();
            if text.starts_with("#-") {
                -1.0
            } else if text.starts_with("#") {
                1.0
            } else {
                let pawns: f32 = text.parse().unwrap();
				let eval = 50.0 + 50.0 * (2.0 / (1.0 + (-0.00368208 * (pawns * 100.0)).exp()) - 1.0);
                eval * 2.0 - 1.0
            }
        };
        let last = self.positions.last_mut().unwrap();
        last.2 = Some(eval);
        //dbg!(q);
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        if key == b"Termination" {
            if value.as_bytes() != b"Normal" {
                self.termination_normal = true;
            }
        } else if key == b"WhiteElo" || key == b"BlackElo" {
            let elo = std::str::from_utf8(value.as_bytes())
                .unwrap()
                .parse::<u16>();
            if let Ok(elo) = elo {
                self.total_elo += elo;
            }
        }
    }
}

pub struct ChessPositionSet(Vec<(Chess, Move, f32)>);

impl ChessPositionSet {
    pub fn new(games: Vec<(u16, Vec<(Chess, Move, Option<f32>)>)>) -> Self {
        let mut positions = Vec::new();
        for game in games {
            for (_, pos) in game.1.into_iter().enumerate() {
                positions.push((
                    pos.0,
                    pos.1,
                    match pos.2 {
                        None => continue,
                        Some(eval) => eval,
                    },
                ));
            }
        }
        dbg!(positions.len());
        Self(positions)
    }
}

impl ExactSizeDataset for ChessPositionSet {
    // board, value, policy, mask
    type Item<'a> = (Vec<f32>, f32, Vec<f32>, Vec<f32>) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        let (pos, mov, eval) = &self.0[index];
        let data = encode_positions(pos);

        // create policy output
        let mut output = vec![0.0; 4608];
        let (plane_idx, rank_idx, file_idx) = move_to_idx(&mov, pos.turn() == Color::Black);
        let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
        output[mov_idx as usize] = 1.0;

        (
            data.into_raw_vec(),
            turn_to_side(pos.turn()) as f32 * eval,
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
