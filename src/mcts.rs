use shakmaty::uci::*;
use shakmaty::{
    fen::Fen,
    zobrist::{Zobrist128, ZobristHash},
    *,
};
use std::collections::HashMap;
use std::hash::Hash;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::process::*;
use vampirc_uci::{parse_one, UciMessage, UciPiece, UciTimeControl};

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
    nodes: HashMap<B, Node<B>>,
    root: B,
    depth: u8,
}

impl<B: Position + Clone + Eq + PartialEq + Hash> MCTSTree<B> {
    pub fn new(root_board: &B, child: &mut Child) -> Self {
        let mut tree = Self {
            root: root_board.clone(),
            nodes: HashMap::new(),
            depth: 0,
        };

        tree.set_root(root_board, child);

        tree
    }

    pub fn set_root(&mut self, root_board: &B, child: &mut Child) {
        if !self.nodes.contains_key(root_board) {
            self.nodes.insert(
                root_board.clone(),
                Node::new(0.0, turn_to_side(root_board.turn())),
            );
        }
        self.root = root_board.clone();
        self.depth = 0;

        let value = self.expand_node(root_board, |board| {
            use vampirc_uci::UciInfoAttribute;

            let stdin = child.stdin.as_mut().unwrap();
            let mut stdout = BufReader::new(child.stdout.as_mut().unwrap());
            stdin
                .write_all(
                    format!(
                        "position fen {}\n",
                        Fen::from_position(board.clone(), EnPassantMode::Legal)
                    )
                    .as_bytes(),
                )
                .expect("Failed to write to stdin");
            stdin
                .write_all("go depth 1\n".as_bytes())
                .expect("Failed to write to stdin");

            let mut last_value = 0.0;
            let mut mov_table = HashMap::new();
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
                        let mut mov_score = 0.0;
                        let mut mov = None;
                        let mut is_best_line = false;
                        for attr in attrs {
                            match attr {
                                UciInfoAttribute::Score { cp, mate, .. } => {
                                    if let Some(cp) = cp {
                                        mov_score = 2.0
                                            * (1.0
                                                / (1.0 + 10.0f32.powf(-(cp as f32 / 100.0) / 4.0)))
                                            - 1.0;
                                    } else if let Some(mate) = mate {
                                        if mate > 0 {
                                            mov_score = 1.0 - (mate.abs() as f32 * 0.01);
                                        } else {
                                            mov_score = -1.0 + (mate.abs() as f32 * 0.01);
                                        }
                                        //break;
                                    }
                                }
                                UciInfoAttribute::MultiPv(pv) => is_best_line = pv == 1,
                                UciInfoAttribute::Pv(moves) => {
                                    let uci = moves[0].to_string();
                                    let uci: Uci = uci.parse().expect("bad pv");
                                    mov = Some(uci.to_move(board).expect("bad pv"));
                                }
                                _ => {}
                            }
                        }

                        if let Some(mov) = mov {
                            mov_table.insert(mov, mov_score);
                            if is_best_line {
                                last_value = mov_score;
                            }
                        }
                    }
                    UciMessage::BestMove { .. } => break,
                    _ => {}
                }
            }

            let mut distr = vec![];
            let mut sum = 0.0;
            for (mov, value) in mov_table {
                let value = value / 2.0 + 0.5;
                distr.push((mov, value));
                sum += value;
            }

            // rescale
            distr.iter_mut().for_each(|d| d.1 /= sum as f32);

            (
                last_value,
                distr
                    .into_iter()
                    .map(|(mov, value)| {
                        let mut board = board.clone();
                        board.play_unchecked(&mov);

                        let turn = board.turn();

                        (board, mov, Node::new(value, turn_to_side(turn)))
                    })
                    .collect::<Vec<_>>(),
            )
        });

        let mut node = self.nodes.get_mut(&root_board).unwrap();
        node.value_sum = value;
        node.visit_count = 1;
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

        self.depth = self.depth.max(search_path.len() as u8);

        let mut board = boards[boards.len() - 2].clone();
        board.play_unchecked(&last_action.unwrap());
        let value = match board.outcome() {
            Some(Outcome::Decisive { winner }) => turn_to_side(winner) as f32,
            Some(Outcome::Draw) => 0.0,
            None => {
                // expand
                let value = self.expand_node(&board, |board| {
                    use vampirc_uci::UciInfoAttribute;

                    let stdin = child.stdin.as_mut().unwrap();
                    let mut stdout = BufReader::new(child.stdout.as_mut().unwrap());
                    stdin
                        .write_all(
                            format!(
                                "position fen {}\n",
                                Fen::from_position(board.clone(), EnPassantMode::Legal)
                            )
                            .as_bytes(),
                        )
                        .expect("Failed to write to stdin");
                    stdin
                        .write_all("go depth 1\n".as_bytes())
                        .expect("Failed to write to stdin");

                    let mut last_value = 0.0;
                    let mut mov_table = HashMap::new();
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
                                let mut mov_score = 0.0;
                                let mut mov = None;
                                let mut is_best_line = false;
                                for attr in attrs {
                                    match attr {
                                        UciInfoAttribute::Score { cp, mate, .. } => {
                                            if let Some(cp) = cp {
                                                mov_score = 2.0
                                                    * (1.0
                                                        / (1.0
                                                            + 10.0f32
                                                                .powf(-(cp as f32 / 100.0) / 4.0)))
                                                    - 1.0;
                                            } else if let Some(mate) = mate {
                                                if mate > 0 {
                                                    mov_score = 1.0 - (mate.abs() as f32 * 0.01);
                                                } else {
                                                    mov_score = -1.0 + (mate.abs() as f32 * 0.01);
                                                }
                                                //break;
                                            }
                                        }
                                        UciInfoAttribute::MultiPv(pv) => is_best_line = pv == 1,
                                        UciInfoAttribute::Pv(moves) => {
                                            let uci = moves[0].to_string();
                                            let uci: Uci = uci.parse().expect("bad pv");
                                            mov = Some(uci.to_move(board).expect("bad pv"));
                                        }
                                        _ => {}
                                    }
                                }

                                if let Some(mov) = mov {
                                    mov_table.insert(mov, mov_score);
                                    if is_best_line {
                                        last_value = mov_score;
                                    }
                                }
                            }
                            UciMessage::BestMove { .. } => break,
                            _ => {}
                        }
                    }

                    let mut distr = vec![];
                    let mut sum = 0.0;
                    for (mov, value) in mov_table {
                        let value = value / 2.0 + 0.5;
                        distr.push((mov, value));
                        sum += value;
                    }

                    // rescale
                    distr.iter_mut().for_each(|d| d.1 /= sum as f32);

                    (
                        last_value,
                        distr
                            .into_iter()
                            .map(|(mov, value)| {
                                let mut board = board.clone();
                                board.play_unchecked(&mov);

                                let turn = board.turn();

                                (board, mov, Node::new(value, turn_to_side(turn)))
                            })
                            .collect::<Vec<_>>(),
                    )
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
        self.nodes.get(&self.root).unwrap().value()
    }

    pub fn get_depth(&self) -> u8 {
        self.depth
    }
}
