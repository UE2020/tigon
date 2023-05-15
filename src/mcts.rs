use std::hash::Hash;
use shakmaty::{*, zobrist::{ZobristHash, Zobrist64}};
use intmap::IntMap;

pub type BoardHash = u64;

fn turn_to_side(color: Color) -> i8 {
	match color {
		Color::White => 1,
		Color::Black => -1,
	}
}

#[derive(Debug, Clone)]
pub struct Node {
	// Don't point at nodes directly
	children: Option<Vec<BoardHash>>,
	visit_count: u32,
	value_sum: f32,
	prior: f32,	
	to_play: i8,
}

impl Node {
	pub fn new(prior: f32, to_play: i8) -> Self {
		Self {
			children: None,
			visit_count: 0,
			value_sum: 0.0,
			prior,
			to_play,
		}
	}

	pub fn expanded(&self) -> bool {
		self.children.is_some()
	}

	pub fn value(&self) -> f32 {
		self.value_sum / self.visit_count as f32
	}

	// pub fn select_child(&self, tree: &MCTSTree) -> (Move, BoardHash) {
		
	// }
}

#[derive(Debug, Clone)]
pub struct MCTSTree {
    nodes: IntMap<Node>,
    root: BoardHash,
}

impl MCTSTree {
    pub fn new<B: Position>(root_board: &B) -> Self {
        let mut tree = Self {
            root: 0,
            nodes: IntMap::new(),
        };

		tree.set_root(root_board);
		
		tree
    }

	pub fn set_root<B: Position>(&mut self, root_board: &B) {
		let hash: Zobrist64 = root_board.zobrist_hash(EnPassantMode::Legal);
		if !self.nodes.contains_key(hash.0) {
			self.nodes.insert(hash.0, Node::new(0.0, turn_to_side(root_board.turn())));
		}
		self.root = hash.0;
	}

	pub fn expand_node<B: Position, P: FnOnce(&B) -> Vec<(BoardHash, Node)>>(&mut self, node: &B, prior: P) {
		let new_nodes = prior(node);
		let hash: Zobrist64 = node.zobrist_hash(EnPassantMode::Legal);
		if let Some(node) = self.nodes.get_mut(hash.0) {
			node.children = Some(new_nodes.iter().map(|(hash, _)| *hash).collect::<Vec<_>>());
			for (hash, node) in new_nodes {
				self.nodes.insert(hash, node);
			}
		} else {
			eprintln!("Warning: node not found - {}", node.board().board_fen(Bitboard(0)));
		}
	}

	pub fn rollout(&mut self) {
		let mut search_path = vec![self.root];
		let mut node = self.root;
		while self.nodes.get(node).expect("node not found").expanded() {

		}
	}

	pub fn ucb(&self, parent: BoardHash, child: BoardHash, c_puct: f32) -> f32 {
		let parent = self.nodes.get(parent).expect("node not found");
		let child = self.nodes.get(child).expect("node not found");

		let prior_score = child.prior * c_puct * (parent.visit_count as f32).sqrt() / (child.visit_count as f32 + 1.0);
		let value_score = if child.visit_count > 0 {
			-child.value()
		} else {
			0.0
		};

		value_score + prior_score
	}
}