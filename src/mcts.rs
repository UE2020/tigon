use intmap::IntMap;
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    *,
};

pub type BoardHash = u64;

pub const C_PUCT: f32 = 1.2;

fn turn_to_side(color: Color) -> i8 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    // Don't point at nodes directly
    children: Option<Vec<(BoardHash, Move)>>,
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
        (self.value_sum / self.visit_count as f32) / 2.0 + 0.5
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
            self.nodes
                .insert(hash.0, Node::new(0.0, turn_to_side(root_board.turn())));
        }
        self.root = hash.0;
    }

    pub fn expand_node<B: Position, P: FnOnce(&B) -> (f32, Vec<(BoardHash, Move, Node)>)>(
        &mut self,
        node: &B,
        prior: P,
    ) -> f32 {
        let (value, new_nodes) = prior(node);
        let hash: Zobrist64 = node.zobrist_hash(EnPassantMode::Legal);
        if let Some(node) = self.nodes.get_mut(hash.0) {
            node.children = Some(
                new_nodes
                    .iter()
                    .map(|(hash, m, _)| (*hash, m.clone()))
                    .collect::<Vec<_>>(),
            );
            for (hash, _, node) in new_nodes {
                self.nodes.insert(hash, node);
            }
        } else {
            eprintln!(
                "Warning: node not found - {}",
                node.board().board_fen(Bitboard(0))
            );
        }

        value
    }

    pub fn rollout<B: Position + Clone>(&mut self, board: B) -> Result<(), PlayError<B>> {
        let mut search_path = vec![self.root];
        let mut boards = vec![board.clone()];
        let mut curr_node = self.root;
        let mut last_action = None;
        while self
            .nodes
            .get(curr_node)
            .expect("node not found")
            .expanded()
        {
            let node = self.nodes.get(curr_node).expect("node not found");
            let child = node
                .children
                .as_ref()
                .unwrap()
                .iter()
                .max_by(|a, b| {
                    self.ucb(curr_node, a.0, C_PUCT)
                        .partial_cmp(&self.ucb(curr_node, b.0, C_PUCT))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            search_path.push(child.0);
            curr_node = child.0;
            last_action = Some(child.1.clone());
            let last = boards.last();
            let mut last = last.unwrap().clone();
            last.play_unchecked(&child.1);
            boards.push(last);
        }

        let parent = search_path[search_path.len() - 2];
        let mut board = boards[boards.len() - 2].clone();
        board.play_unchecked(&last_action.unwrap());
        let value = match board.outcome() {
            Some(Outcome::Decisive { winner }) => turn_to_side(winner) as f32,
            Some(Outcome::Draw) => 0.0,
            None => {
                // expand
                let value = self.expand_node(boards.last().unwrap(), |_| todo!());
                if board.turn() == Color::Black {
                    -value
                } else {
                    value
                }
            }
        };

        // back up
        for node in search_path.iter() {
            let mut node = self.nodes.get_mut(*node).expect("node not found");
            node.visit_count += 1;
            node.value_sum += value * node.to_play as f32;
        }

        Ok(())
    }

    pub fn ucb(&self, parent: BoardHash, child: BoardHash, c_puct: f32) -> f32 {
        let parent = self.nodes.get(parent).expect("node not found");
        let child = self.nodes.get(child).expect("node not found");

        let prior_score = child.prior * c_puct * (parent.visit_count as f32).sqrt()
            / (child.visit_count as f32 + 1.0);
        let value_score = if child.visit_count > 0 {
            -child.value()
        } else {
            0.0
        };

        value_score + prior_score
    }
}
