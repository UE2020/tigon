use std::collections::HashMap as IntMap;
use std::hash::Hash;
use shakmaty::{
    zobrist::{Zobrist128, ZobristHash},
    *, fen::Fen,
};
use vampirc_uci::{parse_one, UciMessage, UciPiece, UciTimeControl};
use std::process::*;
use std::io::prelude::*;

pub const C_PUCT: f32 = 1.2;

fn turn_to_side(color: Color) -> i8 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}

#[derive(Debug, Clone)]
pub struct Node<B> {
    // Don't point at nodes directly
    children: Option<Vec<(B, Move)>>,
    visit_count: u32,
    value_sum: f32,
    prior: f32,
    to_play: i8,
}

impl<B> Node<B> {
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
        (self.value_sum / self.visit_count as f32) / 2.0 + 0.5
    }

    // pub fn select_child(&self, tree: &MCTSTree) -> (Move, B) {

    // }
}

#[derive(Debug, Clone)]
pub struct MCTSTree<B: Position + Clone + Eq + PartialEq + Hash> {
    nodes: IntMap<B, Node<B>>,
    root: B,
}

impl<B: Position + Clone + Eq + PartialEq + Hash> MCTSTree<B> {
    pub fn new(root_board: &B, child: &mut Child) -> Self {
        let mut tree = Self {
            root: root_board.clone(),
            nodes: IntMap::new(),
        };

        tree.set_root(root_board);
        tree.expand_node(root_board, |board| {
            use vampirc_uci::UciInfoAttribute;

            let stdin = child.stdin.as_mut().unwrap();
            let stdout = child.stdout.as_mut().unwrap();
            stdin
                .write_all(
                    format!("position fen {}\n", Fen::from_position(board.clone(), EnPassantMode::Legal))
                        .as_bytes(),
                )
                .expect("Failed to write to stdin");
            stdin
                .write_all("go depth 2\n".as_bytes())
                .expect("Failed to write to stdin");

            let mut last_value = 0.0;
            loop {
                let mut bytes = vec![];
                loop {
                    // read a char
                    let mut output = [0];
                    stdout
                        .read_exact(&mut output)
                        .expect("Failed to read output");
                    if output[0] as char == '\n' {
                        break;
                    }
                    bytes.push(output[0]);
                }
                let output = String::from_utf8_lossy(&bytes);
                let msg = parse_one(&output);
                match msg {
                    UciMessage::Info(attrs) => {
                        for attr in attrs {
                            match attr {
                                UciInfoAttribute::Score { cp, mate, .. } => {
                                    if let Some(cp) = cp {
                                        last_value = 2.0
                                            * (1.0
                                                / (1.0
                                                    + 10.0f32
                                                        .powf(-(cp as f32 / 100.0) / 4.0)))
                                            - 1.0;
                                    } else if let Some(mate) = mate {
                                        if mate > 0 {
                                            last_value = 1.0 - (mate.abs() as f32 * 0.01);
                                        } else {
                                            last_value = -1.0 + (mate.abs() as f32 * 0.01);
                                        }
                                        stdin
                                            .write_all("stop\n".as_bytes())
                                            .expect("Failed to write to stdin");
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    UciMessage::BestMove { .. } => break,
                    _ => {}
                }
            }

            let movs = board.legal_moves();

            (last_value * 2.0 - 1.0, movs.into_iter().map(|m| {
                let mut board = board.clone();
                board.play_unchecked(&m);
                
                let turn = board.turn();

                (board, m, Node::new(0.0, turn_to_side(turn)))
            }).collect::<Vec<_>>())
        });

        tree
    }

    pub fn set_root(&mut self, root_board: &B) {
        if !self.nodes.contains_key(root_board) {
            self.nodes
                .insert(root_board.clone(), Node::new(0.0, turn_to_side(root_board.turn())));
        }
        self.root = root_board.clone();
    }

    pub fn expand_node<P: FnOnce(&B) -> (f32, Vec<(B, Move, Node<B>)>)>(
        &mut self,
        node: &B,
        prior: P,
    ) -> f32 {
        let (value, new_nodes) = prior(node);
        if let Some(node) = self.nodes.get_mut(node) {
            node.children = Some(
                new_nodes
                    .iter()
                    .map(|(board, m, _)| (board.clone(), m.clone()))
                    .collect::<Vec<_>>(),
            );
            for (board, _, node) in new_nodes {
                self.nodes.insert(board, node);
            }
        } else {
            eprintln!(
                "Warning: node not found - {}",
                Fen::from_position(node.clone(), EnPassantMode::Legal)
            );
        }

        value
    }

    pub fn rollout(&mut self, board: B, child: &mut Child) -> Result<(), PlayError<B>> {
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

        let mut board = boards[boards.len() - 2].clone();
        board.play_unchecked(&last_action.unwrap());
        let value = match board.outcome() {
            Some(Outcome::Decisive { winner }) => turn_to_side(winner) as f32,
            Some(Outcome::Draw) => 0.0,
            None => {
                // expand
                let value = self.expand_node(boards.last().unwrap(), |board| {
                    use vampirc_uci::UciInfoAttribute;

                    let stdin = child.stdin.as_mut().unwrap();
                    let stdout = child.stdout.as_mut().unwrap();
                    stdin
                        .write_all(
                            format!("position fen {}\n", Fen::from_position(board.clone(), EnPassantMode::Legal))
                                .as_bytes(),
                        )
                        .expect("Failed to write to stdin");
                    stdin
                        .write_all("go depth 2\n".as_bytes())
                        .expect("Failed to write to stdin");

                    let mut last_value = 0.0;
                    loop {
                        let mut bytes = vec![];
                        loop {
                            // read a char
                            let mut output = [0];
                            stdout
                                .read_exact(&mut output).expect("read failed");
                            if output[0] as char == '\n' {
                                break;
                            }
                            bytes.push(output[0]);
                        }
                        let output = String::from_utf8_lossy(&bytes);
                        let msg = parse_one(&output);
                        match msg {
                            UciMessage::Info(attrs) => {
                                for attr in attrs {
                                    match attr {
                                        UciInfoAttribute::Score { cp, mate, .. } => {
                                            if let Some(cp) = cp {
                                                last_value = 2.0
                                                    * (1.0
                                                        / (1.0
                                                            + 10.0f32
                                                                .powf(-(cp as f32 / 100.0) / 4.0)))
                                                    - 1.0;
                                            } else if let Some(mate) = mate {
                                                if mate > 0 {
                                                    last_value = 1.0 - (mate.abs() as f32 * 0.01);
                                                } else {
                                                    last_value = -1.0 + (mate.abs() as f32 * 0.01);
                                                }
                                                stdin
                                                    .write_all("stop\n".as_bytes())
                                                    .expect("Failed to write to stdin");
                                                break;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            UciMessage::BestMove { .. } => break,
                            _ => {}
                        }
                    }

                    let movs = board.legal_moves();

                    (last_value * 2.0 - 1.0, movs.into_iter().map(|m| {
                        let mut board = board.clone();
                        board.play_unchecked(&m);
                        
                        let turn = board.turn();
        
                        (board, m, Node::new(0.0, turn_to_side(turn)))
                    }).collect::<Vec<_>>())
                });

                if board.turn() == Color::Black {
                    -value
                } else {
                    value
                }
            }
        };

        // back up
        for node in search_path.iter() {
            let mut node = self.nodes.get_mut(node).expect("node not found");
            node.visit_count += 1;
            node.value_sum += value * node.to_play as f32;
        }

        Ok(())
    }

    pub fn ucb(&self, parent: &B, child: &B, c_puct: f32) -> f32 {
        let parent = self.nodes.get(&parent).expect("node not found");
        let child = self.nodes.get(&child).expect("node not found");

        let prior_score = child.prior * c_puct * (parent.visit_count as f32).sqrt()
            / (child.visit_count as f32 + 1.0);
        let value_score = if child.visit_count > 0 {
            -child.value()
        } else {
            0.0
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
                    self.ucb(&curr_node,& a.0, C_PUCT)
                        .partial_cmp(&self.ucb(&curr_node,& b.0, C_PUCT))
                        .unwrap_or(std::cmp::Ordering::Equal)
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
}
