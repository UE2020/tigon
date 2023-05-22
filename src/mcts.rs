use crate::*;

use shakmaty::uci::*;
use shakmaty::{
    fen::Fen,
    *,
};
use std::collections::HashMap;
use std::hash::Hash;
use std::io::prelude::*;
use std::io::BufReader;
use std::process::*;
use vampirc_uci::{parse_one, UciMessage};

pub const C_PUCT: f32 = 1.5;

pub fn turn_to_side(color: Color) -> i8 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}

#[derive(Debug, Clone)]
pub struct Node<B> {
    // Don't point at nodes directly
    children: Option<Vec<(B, Move)>>,
	parents: Vec<B>,
    visit_count: u32,
    value_sum: f32,
    prior: f32,
    to_play: i8,
}

impl<B> Node<B> {
    pub fn new(prior: f32, to_play: i8, creator: Option<B>) -> Self {
        Self {
            children: None,
			parents: if let Some(creator) = creator {
				vec![creator]
			} else {
				vec![]
			},
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
}

#[derive(Debug, Clone)]
pub struct MCTSTree<B: Position + Clone + Eq + PartialEq + Hash> {
    nodes: HashMap<B, Node<B>>,
    root: B,
    depth: u8,
}

impl<B: Position + Clone + Eq + PartialEq + Hash> MCTSTree<B> {
    pub fn new<P: FnOnce(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(root_board: &B, model: P) -> Self {
        let mut tree = Self {
            root: root_board.clone(),
            nodes: HashMap::new(),
            depth: 0,
        };

        tree.set_root(root_board, model);

        tree
    }

    pub fn set_root<P: FnOnce(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(&mut self, root_board: &B, model: P) {
        if !self.nodes.contains_key(root_board) {
            self.nodes.insert(
                root_board.clone(),
                Node::new(0.0, turn_to_side(root_board.turn()), None),
            );
        }
        self.root = root_board.clone();
        self.depth = 0;

        let node = self.nodes.get(&root_board);
		let (expanded, value) = if let Some(node) = node {
			if !node.expanded() {
				(true, self.expand_node(&root_board, model))
			} else {
				(false, 0.0)
			}
		} else {
			unreachable!()
		};

		if expanded {
			let node = self.nodes.get_mut(&root_board).unwrap();

			node.visit_count = 1;
			node.value_sum = value;
		}
    }

    pub fn expand_node<P: FnOnce(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        node: &B,
        prior: P,
    ) -> f32 {
        let (value, new_nodes) = prior(node);
        if let Some(expanding_node) = self.nodes.get_mut(node) {
            expanding_node.children = Some(
                new_nodes
                    .iter()
                    .map(|(board, m, _)| (board.clone(), m.clone()))
                    .collect::<Vec<_>>(),
            );
            for (board, _, new_node) in new_nodes {
				if let Some(new_node) = self.nodes.get_mut(&board) {
					new_node.parents.push(node.clone());
				} else {
					self.nodes.insert(board, new_node);
				}
            }
        } else {
            eprintln!(
                "Warning: node not found - {}",
                Fen::from_position(node.clone(), EnPassantMode::Legal)
            );
        }

        value
    }

    pub fn rollout<P: FnOnce(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(&mut self, board: B, model: P) -> Result<(), PlayError<B>> {
        let mut search_path = vec![self.root.clone()];
        let mut boards = vec![board.clone()];
        let mut curr_node = self.root.clone();
        let mut last_action = None;
        while self
            .nodes
            .get(&curr_node)
            .expect("node not found")
            .expanded()
        {
            let node = self.nodes.get(&curr_node).expect("node not found");
            let child = node
                .children
                .as_ref()
                .unwrap()
                .iter()
                .max_by(|a, b| {
                    self.ucb(&curr_node, &a.0, C_PUCT)
                        .partial_cmp(&self.ucb(&curr_node, &b.0, C_PUCT))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            search_path.push(child.0.clone());
            curr_node = child.0.clone();
            last_action = Some(child.1.clone());
            let last = boards.last();
            let mut last = last.unwrap().clone();
            last.play_unchecked(&child.1);
            boards.push(last);
        }

        self.depth = self.depth.max(search_path.len() as u8);

        let mut board = boards[boards.len() - 2].clone();
        board.play_unchecked(&last_action.unwrap());
        let value = match board.outcome() {
            Some(Outcome::Decisive { winner }) => turn_to_side(winner) as f32,
            Some(Outcome::Draw) => 0.0,
            None => {
                // expand
                let value = self.expand_node(&board, model);

                if board.turn() == Color::Black {
                    -value
                } else {
                    value
                }
            }
        };

		self.backpropogate(&board, value);

        // back up
        // for node in search_path.iter() {
        //     let mut node = self.nodes.get_mut(node).expect("node not found");
        //     node.visit_count += 1;
        //     node.value_sum += value * node.to_play as f32;
        // }

        Ok(())
    }

	pub fn backpropogate(&mut self, board: &B, value: f32) {
		let node = self.nodes.get_mut(&board).unwrap();
		node.visit_count += 1;
		node.value_sum += value * node.to_play as f32;
		let parents = node.parents.clone();
		for parent in parents {
			self.backpropogate(&parent, value);
		}
	}

    pub fn ucb(&self, parent: &B, child: &B, c_puct: f32) -> f32 {
        let parent = self.nodes.get(&parent).expect("node not found");
        let child = self.nodes.get(&child).expect("node not found");

        let prior_score = child.prior * c_puct * (parent.visit_count as f32).sqrt()
            / (child.visit_count as f32 + 1.0);
        let value_score = if child.visit_count > 0 {
            (-child.value()) / 2.0 + 0.5
        } else {
            0.5
        };

        value_score + prior_score
    }

    pub fn pv(&self) -> Vec<Move> {
        let mut pv = vec![];
        let mut curr_node = self.root.clone();
        while self
            .nodes
            .get(&curr_node)
            .expect("node not found")
            .expanded()
        {
            let node = self.nodes.get(&curr_node).expect("node not found");
            let child = node
                .children
                .as_ref()
                .unwrap()
                .iter()
                .max_by(|a, b| {
                    let a_visits = self.nodes.get(&a.0).unwrap().visit_count;
                    let b_visits = self.nodes.get(&b.0).unwrap().visit_count;
                    a_visits.cmp(&b_visits)
                })
                .unwrap();
            curr_node = child.0.clone();
            pv.push(child.1.clone());
        }

        pv
    }

    pub fn root_distribution(&self, root: &B) -> Vec<(Move, f32)> {
        let moves = root.legal_moves();
        let mut distr = vec![];
        let mut sum = 0;
        for mov in moves {
            let mut child = root.clone();
            child.play_unchecked(&mov);
            let child_node = self.nodes.get(&child).expect("node not found");
            distr.push((mov, child_node.visit_count as f32));
            sum += child_node.visit_count;
        }

        // rescale
        distr.iter_mut().for_each(|d| d.1 /= sum as f32);

        distr
    }

    pub fn get_root_q(&self) -> f32 {
        self.nodes.get(&self.root).unwrap().value() / 2.0 + 0.5
    }

    pub fn get_depth(&self) -> u8 {
        self.depth
    }
}
