use dfdx::prelude::*;
use shakmaty::fen::*;
use shakmaty::uci::*;
use shakmaty::*;
use std::io::Write;
use std::process::{Command, Stdio};

use std::io::{self, BufRead};
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};
use std::time::{Duration, Instant};

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

    child
        .stdin
        .as_ref()
        .unwrap()
        .write_all(b"setoption name MultiPV value 256")
        .expect("failed to set MultiPV");

    let mut pos: Chess = Chess::default();
    let mut mcts = mcts::MCTSTree::new(&pos, &mut child);

    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let msg: UciMessage = parse_one(&line);
        match msg {
            UciMessage::Uci => {
                println!("id name ProphetNNUE");
                println!("uciok")
            }
            UciMessage::Position {
                startpos,
                moves,
                fen,
            } => {
                dbg!(&fen);
                if startpos {
                    pos = Chess::default();
                } else if let Some(fen) = fen {
                    pos = Fen::from_ascii(fen.0.as_bytes())
                        .expect("bad fen")
                        .into_position(CastlingMode::Standard)
                        .expect("bad fen");
                }

                for mov in moves {
                    let uci = mov.to_string();
                    let uci: Uci = uci.parse().expect("bad uci");
                    let m = uci.to_move(&pos).expect("bad move");
                    pos.play_unchecked(&m);
                }

                mcts.set_root(&pos, &mut child);
            }
            UciMessage::Go { time_control, .. } => {
				let target = match time_control {
                    Some(time) => match time {
                        UciTimeControl::MoveTime(duration) => duration.to_std().unwrap(),
                        UciTimeControl::TimeLeft {
                            white_time,
                            black_time,
                            white_increment,
                            ..
                        } => {
                            let time_left = match pos.turn() {
                                Color::White => {
                                    white_time.unwrap().to_std().unwrap()
                                }
                                Color::Black => {
                                    black_time.unwrap().to_std().unwrap()
                                }
                            };

                            (time_left / 40).min(Duration::from_secs(60))
                                + white_increment.unwrap_or(vampirc_uci::Duration::milliseconds(0)).to_std().unwrap()
                        }
                        _ => Duration::from_millis(60000),
                    },
                    None => Duration::from_millis(60000),
                };

				let now = Instant::now();
                for i in 0..10000 {
                    mcts.rollout(pos.clone(), &mut child).unwrap();
                    
					if i % 800 == 0 {
						println!(
							"info depth {} score cp {} nodes {} nps {} pv {}",
							mcts.get_depth(),
							(290.680623072 * (3.096181612 * (mcts.get_root_q() - 0.5)).tan() / 10.0 / 1.5) as i32,
							i,
							(i as f32 / now.elapsed().as_secs_f32()) as u32,
							mcts.pv()
								.iter()
								.map(|m| m.to_uci(pos.castles().mode()).to_string())
								.collect::<Vec<_>>()
								.join(" ")
						);
					}

					if now.elapsed() >= target {
                        break;
                    }
                }

                let pv = mcts.pv();
                println!(
                    "bestmove {}",
                    pv[0].to_uci(pos.castles().mode()).to_string()
                );

                // for (mov, prob) in mcts.root_distribution(&pos) {
                // 	println!(
                // 		"{}: {:.2}%",
                // 		mov.to_uci(pos.castles().mode()).to_string(),
                // 		prob * 100.0
                // 	);
                // }

                //let result = search::iterative_deepening_search(board, &dev, &mut nnue);
                //println!("bestmove {}", result.0);
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            _ => {}
        }
    }

    Ok(())

    /*let dev = AutoDevice::default();
    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();*/
}
