use crate::*;

use shakmaty::fen::Fen;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroU32;
use std::sync::Arc;

pub const C_PUCT: f32 = 1.75;

pub fn turn_to_side(color: Color) -> i8 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}

#[derive(Debug, Clone)]
pub struct ParentPointer<B> {
    pos: B,
    prior: f32,
}

impl<B> ParentPointer<B> {
    pub fn new(pos: B, prior: f32) -> ParentPointer<B> {
        Self { pos, prior }
    }
}

#[derive(Debug, Clone)]
pub struct ChildPointer<B> {
    pos: B,
    mov: Move,
    visits: u32,
}

impl<B> ChildPointer<B> {
    pub fn new(pos: B, mov: Move) -> ChildPointer<B> {
        Self {
            pos,
            mov,
            visits: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node<B> {
    // Don't point at nodes directly
    children: Option<Vec<RefCell<ChildPointer<B>>>>,
    parents: Vec<ParentPointer<B>>,
    visit_count: u32,
    value: f32,
    nn_value: f32,
    to_play: i8,
}

impl<B> Node<B> {
    pub fn new(value: f32, prior: f32, to_play: i8, creator: Option<B>) -> Self {
        Self {
            children: None,
            parents: if let Some(creator) = creator {
                vec![ParentPointer::new(creator, prior)]
            } else {
                vec![]
            },
            visit_count: 0,
            value,
            nn_value: value,
            to_play,
        }
    }

    pub fn expanded(&self) -> bool {
        self.children.is_some()
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

pub type HistoryTable<B> = HashMap<B, u16>;

#[derive(Debug, Clone)]
pub struct MCTSTree<B: Position + Syzygy + Clone + Eq + PartialEq + Hash + std::fmt::Debug> {
    nodes: HashMap<B, Node<B>>,
    root: B,
    root_ply_counter: NonZeroU32,
    depth: u8,
}

impl<B: Position + Syzygy + Clone + Eq + PartialEq + Hash + std::fmt::Debug> MCTSTree<B> {
    pub fn new<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(root_pos: &B, model: Arc<P>) -> Self {
        let mut tree = Self {
            root: root_pos.clone(),
            root_ply_counter: root_pos.fullmoves(),
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
        let full_moves = root_pos.fullmoves();
        self.root_ply_counter = full_moves;
        if !self.nodes.contains_key(root_pos) {
            self.nodes.insert(
                root_pos.clone(),
                Node::new(0.0, 0.0, turn_to_side(root_pos.turn()), None),
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
            node.value = value;
        }
    }

    pub fn expand_node<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        node: &B,
        prior: Arc<P>,
    ) -> f32 {
        let (value, new_nodes) = prior(node);
        if let Some(expanding_node) = self.nodes.get_mut(node) {
            assert!(!expanding_node.children.is_some());
            expanding_node.children = Some(
                new_nodes
                    .iter()
                    .map(|(pos, m, _)| RefCell::new(ChildPointer::new(pos.clone(), m.clone())))
                    .collect::<Vec<_>>(),
            );
            for (pos, _, new_node) in new_nodes {
                if let Some(extant_node) = self.nodes.get_mut(&pos) {
                    extant_node
                        .parents
                        .push(ParentPointer::new(node.clone(), new_node.parents[0].prior));
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

	// position fen r5k1/pp4pp/1n2pp2/8/1Q2P3/Pb1rBP2/2q1BKPP/RR6 w - - 14 31 moves b1c1 c2b2 a1b1 b2a2 b1a1 a2b2 a1b1 b2a2 
    pub fn rollout<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        tablebase: Option<&Tablebase<B>>,
        model: Arc<P>,
        tbhits: &mut usize,
		mut history: HistoryTable<Board>,
    ) -> Result<(), PlayError<B>> {
        let mut search_path = vec![self.root.clone()];
        let mut curr_node = self.root.clone();
		let mut is_repetition = false;
        while self
            .nodes
            .get(&curr_node)
            .expect("node not found")
            .expanded()
        {
            let node = self.nodes.get(&curr_node).expect("node not found");
            let mut child = node
                .children
                .as_ref()
                .unwrap()
                .iter()
                .max_by(|a, b| {
                    self.ucb(&curr_node, &a.borrow().pos, C_PUCT)
                        .partial_cmp(&self.ucb(&curr_node, &b.borrow().pos, C_PUCT))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
                .borrow_mut();
            child.visits += 1;
            search_path.push(child.pos.clone());
            curr_node = child.pos.clone();
			history.entry(child.pos.board().clone()).and_modify(|counter| *counter += 1).or_insert(1);
			if let Some(cnt) = history.get(&search_path.last().unwrap().board()) {
				if *cnt >= 2 {
					is_repetition = true;
					break;
				}
			}
        }

        self.depth = self.depth.max(search_path.len() as u8);
        let pos = search_path.last().unwrap();
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
                let value = match (tablebase_result, is_repetition) {
					(_, true) => {
						// unexpand (?) node
						self.nodes.get_mut(&pos).unwrap().children = None;
						0.0
					},
                    (None, _) => self.expand_node(&pos, model),
                    (Some(tb), _) => tb,
                };

                if pos.turn() == Color::Black {
                    -value
                } else {
                    value
                }
            }
        };

        // begin by propogating the value to the node at the tip
        let node = self.nodes.get_mut(&pos).unwrap();
        node.nn_value = value * node.to_play as f32;

        for pos in search_path.into_iter().rev() {
            let node = self.nodes.get(&pos).unwrap();
            let mut values = vec![(1, node.nn_value)];
            let mut total_visits = 1;
            if let Some(children) = node.children.as_ref() {
                for child in children {
                    let child = child.borrow();
                    let child_node = self.nodes.get(&child.pos);
                    if let Some(child_node) = child_node {
                        if child_node.visit_count < 1 {
                            continue;
                        }
                        values.push((child.visits, -child_node.value));
                        total_visits += child.visits;
                    }
                }
            }

            let node = self.nodes.get_mut(&pos).unwrap();
            let value: f32 = values
                .into_iter()
                .map(|(visits, value)| (visits as f32 / total_visits as f32) * value)
                .sum();
            node.visit_count += 1;
            node.value = value;
        }

        Ok(())
    }

    pub fn ucb(&self, parent: &B, child: &B, c_puct: f32) -> f32 {
        let parent_node = self.nodes.get(&parent).expect("node not found");
        let child_node = self.nodes.get(&child).expect("node not found");
        let child_ref = parent_node
            .children
            .as_ref()
            .unwrap()
            .iter()
            .find(|p| p.borrow().pos == *child)
            .expect("no child found")
            .borrow();
        let parent_ref = child_node
            .parents
            .iter()
            .find(|p| p.pos == *parent)
            .expect("no parent found");
        let base = 38739.0;
        let factor = 3.894;
        let final_cpuct = c_puct + factor + ((child_ref.visits as f32 + base) / base).ln();
        let prior_score = parent_ref.prior * final_cpuct * (parent_node.visit_count as f32).sqrt()
            / (child_ref.visits as f32 + 1.0);
        let value_score = if child_node.visit_count > 0 {
            (-child_node.value()) / 2.0 + 0.5
        } else {
            0.0
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
                    let a_visits = a.borrow().visits;
                    let b_visits = b.borrow().visits;
                    a_visits.cmp(&b_visits)
                })
                .unwrap();
            curr_node = child.borrow().pos.clone();
            pv.push(child.borrow().mov.clone());
        }

        pv
    }

    pub fn pv(&self) -> Vec<Move> {
        self.pv_from_node(self.root.clone())
    }

    pub fn all_pvs(&self) -> Vec<(u32, f32, Vec<Move>)> {
        let root_node = self.nodes.get(&self.root).expect("no root");
        let mut pvs = vec![];
        for child_ptr in root_node
            .children
            .as_ref()
            .expect("no children at root")
            .iter()
        {
            if child_ptr.borrow().visits == 0 {
                continue;
            }
            let mut new_pv = vec![child_ptr.borrow().mov.clone()];
            new_pv.extend_from_slice(&self.pv_from_node(child_ptr.borrow().pos.clone()));
            let child = self.nodes.get(&child_ptr.borrow().pos).unwrap();
            pvs.push((child.visit_count, (-child.value()) / 2.0 + 0.5, new_pv));
        }

        pvs
    }

    pub fn root_distribution(&self) -> Vec<(Move, f32)> {
        let mut distr = vec![];
        let mut sum = 0;
        let root = self.nodes.get(&self.root).unwrap();
        for child in root.children.as_ref().unwrap() {
            let child = child.borrow();
            distr.push((child.mov.clone(), child.visits as f32));
            sum += child.visits;
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

    pub fn clear<P: Fn(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(&mut self, model: Arc<P>) {
        *self = Self::new(&self.root, model);
    }
}

pub fn q_to_cp(q: f32) -> i32 {
    let q = q * 2.0 - 1.0;
    (-(q.signum() * (1.0 - q.abs()).ln() / (1.2f32).ln()) * 100.0 / 2.0) as i32
}
