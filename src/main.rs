use dfdx::prelude::*;
use std::process::{Command, Stdio};
use shakmaty::*;

pub mod mcts;

type Mlp = (
    (Linear<5, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() -> Result<(), PlayError<Chess>> {
    let mut child = Command::new("stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to execute child");

    let pos = Chess::default();
    let pos = pos.play(&Move::Normal {
        role: Role::Pawn,
        from: shakmaty::Square::D2,
        to: shakmaty::Square::D4,
        capture: None,
        promotion: None,
    })?;
    let mut mcts = mcts::MCTSTree::new(&pos, &mut child);
    for i in 0..100000 {
        mcts.rollout(pos.clone(), &mut child).unwrap();
        println!("Rollout {}", i);
    }

    println!("{}", mcts.pv().iter().map(|m| m.to_uci(pos.castles().mode()).to_string()).collect::<Vec<_>>().join(" "));
    for (mov, prob) in mcts.root_distribution(&pos) {
        println!("{}: {:.2}%", mov.to_uci(pos.castles().mode()).to_string(), prob * 100.0);
    }

    Ok(())
    
    /*let dev = AutoDevice::default();
    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();*/
}
