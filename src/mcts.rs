use crate::*;

use shakmaty::uci::*;
use shakmaty::{fen::Fen, *};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::io::prelude::*;
use std::io::BufReader;
use std::process::*;
use std::sync::Arc;
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
pub struct MCTSTree<B: Position + Syzygy + Clone + Eq + PartialEq + Hash> {
    nodes: HashMap<B, Node<B>>,
    root: B,
    root_ply_counter: u32,
    depth: u8,
}

impl<B: Position + Syzygy + Clone + Eq + PartialEq + Hash> MCTSTree<B> {
    pub fn new<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(root_pos: &B, model: Arc<P>) -> Self {
        let mut tree = Self {
            root: root_pos.clone(),
            root_ply_counter: root_pos.halfmoves(),
            nodes: HashMap::new(),
            depth: 0,
        };

        tree.set_root(root_pos, model);

        tree
    }

    pub fn set_root<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        root_pos: &B,
        model: Arc<P>,
    ) {
        self.root_ply_counter = root_pos.halfmoves();
        if !self.nodes.contains_key(root_pos) {
            self.nodes.insert(
                root_pos.clone(),
                Node::new(0.0, turn_to_side(root_pos.turn()), None),
            );
        }
        self.root = root_pos.clone();
        self.depth = 0;

        let node = self.nodes.get(&root_pos);
        let (expanded, value) = if let Some(node) = node {
            if !node.expanded() {
                (true, self.expand_node(&root_pos, model))
            } else {
                (false, 0.0)
            }
        } else {
            unreachable!()
        };

        let node = self.nodes.get_mut(&root_pos).unwrap();
        if expanded {
            node.visit_count = 1;
            node.value_sum = value;
        }
    }

    pub fn expand_node<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        node: &B,
        prior: Arc<P>,
    ) -> f32 {
        let (value, new_nodes) = prior(node);
        if let Some(expanding_node) = self.nodes.get_mut(node) {
            expanding_node.children = Some(
                new_nodes
                    .iter()
                    .map(|(pos, m, _)| (pos.clone(), m.clone()))
                    .collect::<Vec<_>>(),
            );
            for (pos, _, new_node) in new_nodes {
                if let Some(new_node) = self.nodes.get_mut(&pos) {
                    new_node.parents.push(node.clone());
                } else {
                    self.nodes.insert(pos, new_node);
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

    pub fn rollout<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        pos: B,
        tablebase: Option<&Tablebase<B>>,
        model: Arc<P>,
        tbhits: &mut usize,
    ) -> Result<(), PlayError<B>> {
        let mut search_path = vec![self.root.clone()];
        let mut positions = vec![pos.clone()];
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
            let last = positions.last();
            let mut last = last.unwrap().clone();
            last.play_unchecked(&child.1);
            positions.push(last);
        }

        self.depth = self.depth.max(search_path.len() as u8);

        let mut pos = positions[positions.len() - 2].clone();
        pos.play_unchecked(&last_action.unwrap());
        let value = match pos.outcome() {
            Some(Outcome::Decisive { winner }) => turn_to_side(winner) as f32,
            Some(Outcome::Draw) => 0.0,
            None => {
                // first, try the tablebase
                let tablebase_result = match tablebase {
                    Some(tablebase) => {
                        if (pos.board().black() | pos.board().white()).count()
                            <= tablebase.max_pieces()
                        {
                            let wdl = tablebase.probe_wdl(&pos);
                            if let Ok(wdl) = wdl {
                                *tbhits += 1;
                                Some(wdl.signum() as f32)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    None => None,
                };

                // expand
                let value = match tablebase_result {
                    None => self.expand_node(&pos, model),
                    Some(tb) => tb,
                };

                if pos.turn() == Color::Black {
                    -value
                } else {
                    value
                }
            }
        };

        let mut collision_table = HashSet::new();
        self.backpropogate(&pos, value, &mut collision_table);

        Ok(())
    }

    pub fn backpropogate(&mut self, pos: &B, value: f32, table: &mut HashSet<B>) {
        // if a node has been seen, its parents must have been seen too
        // don't backpropogate all the way to the root of the tree
        if table.contains(pos) || pos.halfmoves() < self.root_ply_counter {
            return;
        }
        let node = self.nodes.get_mut(&pos).unwrap();
        node.visit_count += 1;
        node.value_sum += value * node.to_play as f32;
        table.insert(pos.clone());
        let parents = node.parents.clone();
        for parent in parents {
            self.backpropogate(&parent, value, table);
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

    pub fn pv_from_node(&self, pos: B) -> Vec<Move> {
        let mut pv = vec![];
        let mut curr_node = pos;
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

    pub fn pv(&self) -> Vec<Move> {
        self.pv_from_node(self.root.clone())
    }

    pub fn all_pvs(&self) -> Vec<(u32, f32, Vec<Move>)> {
        let root_node = self.nodes.get(&self.root).expect("no root");
        let mut pvs = vec![];
        for (child, mov) in root_node
            .children
            .as_ref()
            .expect("no children at root")
            .iter()
        {
            let mut new_pv = vec![mov.clone()];
            new_pv.extend_from_slice(&self.pv_from_node(child.clone()));
            let child = self.nodes.get(child).unwrap();
            pvs.push((child.visit_count, (-child.value()) / 2.0 + 0.5, new_pv));
        }

        pvs
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
        self.nodes.get(&self.root).unwrap().value()
    }

    pub fn get_depth(&self) -> u8 {
        self.depth
    }

    pub fn total_size(&self) -> usize {
        self.nodes.len()
    }
}
