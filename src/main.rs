use shakmaty::fen::*;
use shakmaty::uci::*;
use shakmaty::variant::*;
use shakmaty::*;
use std::io::Write;
use std::process::{Command, Stdio};

use std::io::{self, BufRead};
use std::time::{Duration, Instant};
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};

pub mod mcts;
pub mod encoding;

fn main() -> Result<(), PlayError<Chess>> {
    let mut model = tch::CModule::load("Net_10x128.pt").expect("model is in path");
    model.set_eval();

	let inference = |pos: &Chess| {
		let (policy, value) = encoding::get_neural_output(pos, &model);

		(value, policy.into_iter().map(|(mov, prior)| {
			let mut new_pos = pos.clone();
			new_pos.play_unchecked(&mov);

			let turn = new_pos.turn();

			(new_pos, mov, mcts::Node::new(prior, mcts::turn_to_side(turn), Some(pos.clone())))
		}).collect::<Vec<_>>())
	};

    let mut pos: Chess = Chess::default();
    let mut mcts = mcts::MCTSTree::new(&pos, inference);
	let mut castling_mode = CastlingMode::Standard;
	println!("Tigon ready");

    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let msg: UciMessage = parse_one(&line);
        match msg {
            UciMessage::Uci => {
                println!("id name ProphetNNUE");
				println!("option name UCI_Variant type string default <empty>");
				println!("option name UCI_Chess960 type check default false");
                println!("uciok")
            },
			UciMessage::SetOption { name, value } => {
				if name == "UCI_Chess960" {
					if &value.unwrap() == "true" {
						castling_mode = CastlingMode::Chess960;
						println!("Using chess960");
					} else {
						castling_mode = CastlingMode::Standard;
					}
				}
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
                        .into_position(castling_mode)
                        .expect("bad fen");
                }

                for mov in moves {
                    let uci = mov.to_string();
                    let uci: Uci = uci.parse().expect("bad uci");
                    let m = uci.to_move(&pos).expect("bad move");
                    pos.play_unchecked(&m);
                }

                mcts.set_root(&pos, inference);
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
                                Color::White => white_time.unwrap().to_std().unwrap(),
                                Color::Black => black_time.unwrap().to_std().unwrap(),
                            };

                            (time_left / 40).min(Duration::from_secs(60))
                                + white_increment
                                    .unwrap_or(vampirc_uci::Duration::milliseconds(0))
                                    .to_std()
                                    .unwrap()
                        }
                        _ => Duration::from_millis(60000),
                    },
                    None => Duration::from_millis(60000),
                };

                let now = Instant::now();
                let mut last_pv = None;
				tch::no_grad(|| {
					for i in 0..10000 {
						mcts.rollout(pos.clone(), inference).unwrap();
						let current_pv = mcts
							.pv()
							.iter()
							.map(|m| m.to_uci(pos.castles().mode()).to_string())
							.collect::<Vec<_>>()
							.join(" ");
						let passed = now.elapsed() >= target;
						if last_pv.as_ref() != Some(&current_pv) || passed {
							println!(
								"info depth {} score cp {} nodes {} nps {} pv {}",
								mcts.get_depth(),
								((4.0 * ((mcts.get_root_q()) / (1.0 - mcts.get_root_q())).log10()) * 100.0) as i32,
								i,
								(i as f32 / now.elapsed().as_secs_f32()) as u32,
								current_pv
							);
	
							last_pv = Some(current_pv);
						}
	
						if passed {
							break;
						}
					}
				});

				let mut distr = mcts.root_distribution(&pos);
                distr.sort_by(|b, a| a.1.partial_cmp(&b.1).unwrap());
                for (mov, prob) in distr {
                    println!(
                        "info string {}: {:.2}%",
                        mov.to_uci(pos.castles().mode()).to_string(),
                        prob * 100.0
                    );
                }

                let pv = mcts.pv();
                println!(
                    "bestmove {}",
                    pv[0].to_uci(pos.castles().mode()).to_string()
                );

                //let result = search::iterative_deepening_search(board, &dev, &mut nnue);
                //println!("bestmove {}", result.0);
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            c => println!("error: {}", c),
        }
    }

    Ok(())

    /*let dev = AutoDevice::default();
    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();*/
}