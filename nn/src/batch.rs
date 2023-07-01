use burn::{
    data::{dataloader::batcher::Batcher},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

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

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AverageOutcome {
    sum: f32,
    total: u32,
}

impl AverageOutcome {
    pub fn new() -> Self {
        Self { sum: 0.0, total: 0 }
    }

    pub fn add_outcome(&mut self, outcome: Outcome) {
        self.sum += match outcome {
            Outcome::Decisive { winner } => turn_to_side(winner) as f32,
            Outcome::Draw => 0.0,
        };
        self.total += 1;
    }

    pub fn as_value(&self) -> f32 {
        self.sum / self.total as f32
    }
}

#[derive(Debug, Clone)]
pub struct MoveDistribution {
    moves: HashMap<Move, u32>,
    total: u32,
}

impl MoveDistribution {
    pub fn new() -> Self {
        Self {
            moves: HashMap::new(),
            total: 0,
        }
    }

    pub fn add_move(&mut self, mov: &Move) {
        self.moves
            .entry(mov.clone())
            .and_modify(|counter| *counter += 1)
            .or_insert(1);
        self.total += 1;
    }

    pub fn as_probability_vector(&self, pos: &Chess) -> Vec<f32> {
        let mut output = vec![0.0; 4608];
        for (mov, total) in self.moves.iter() {
            let (plane_idx, rank_idx, file_idx) = move_to_idx(&mov, pos.turn() == Color::Black);
            let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
            output[mov_idx as usize] = *total as f32 / self.total as f32;
        }

        output
    }
}

pub type PositionItem = (Chess, (AverageOutcome, MoveDistribution));

pub struct PositionBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct PositionBatch<B: Backend> {
    pub positions: Tensor<B, 4>,
    pub value_targets: Tensor<B, 2>,
	pub policy_tagets: Tensor<B, 2>,
}

impl<B: Backend> PositionBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<PositionItem, PositionBatch<B>> for PositionBatcher<B> {
    fn batch(&self, items: Vec<PositionItem>) -> PositionBatch<B> {
        let images = items
            .iter()
            .map(|(pos, _)| Data::<f32, 1>::from(encode_positions(pos).into_raw_vec().as_slice()))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 22, 8, 8]))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        PositionBatch { positions, targets }
    }
}

pub fn turn_to_side(color: Color) -> i8 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}
