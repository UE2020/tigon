use shakmaty::fen::*;
use shakmaty::san::*;
use shakmaty::uci::*;
use shakmaty::variant::*;
use shakmaty::*;
use std::io::Write;
use std::process::{Command, Stdio};

use shakmaty_syzygy::{Dtz, MaybeRounded, Syzygy, Tablebase, Wdl};
use std::io::{self, BufRead};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};

pub mod encoding;
pub mod mcts;

fn main() -> Result<(), PlayError<Chess>> {
    tch::set_num_threads(4);
    let mut model = tch::CModule::load("Net_10x128.pt").expect("model is in path");
    model.set_eval();
    let model = Arc::new(model);

    let mut tables = Tablebase::new();
    tables.add_directory("./tables").expect("tables not found");

    let inference = {
        let model = model.clone();
        Arc::new(move |pos: &Chess| {
            let (policy, value) = encoding::get_neural_output(pos, &model);

            (
                value,
                policy
                    .into_iter()
                    .map(|(mov, prior)| {
                        let mut new_pos = pos.clone();
                        new_pos.play_unchecked(&mov);

                        let turn = new_pos.turn();

                        (
                            new_pos,
                            mov,
                            mcts::Node::new(prior, mcts::turn_to_side(turn), Some(pos.clone())),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
    };

    let mut pos: Chess = Chess::default();
    let mcts = Arc::new(Mutex::new(mcts::MCTSTree::new(&pos, inference.clone())));
    let mut castling_mode = CastlingMode::Standard;

    let (tx, rx) = mpsc::channel();
    let should_stop = Arc::new(AtomicBool::new(false));
    {
        let should_stop = should_stop.clone();
        let mcts = mcts.clone();
        let inference = inference.clone();
        thread::spawn(move || loop {
            let recv: (Chess, Option<UciTimeControl>) = rx.recv().unwrap();
            should_stop.store(false, Ordering::Relaxed);
            let pos = recv.0;
            let time_control = recv.1;

            let mut mcts = mcts.lock().unwrap();

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
            let mut tbhits = 0;
            tch::no_grad(|| {
                for i in 0..10000 {
                    mcts.rollout(pos.clone(), Some(&tables), inference.clone(), &mut tbhits)
                        .unwrap();
                    if should_stop.load(Ordering::Relaxed) {
                        should_stop.store(false, Ordering::Relaxed);
                        break;
                    }
                    let current_pv = mcts
                        .pv()
                        .iter()
                        .map(|m| m.to_uci(pos.castles().mode()).to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let passed = now.elapsed() >= target;
                    if last_pv.as_ref() != Some(&current_pv) || passed {
                        println!(
                            "info depth {} score cp {} nodes {} nps {} hashfull {} tbhits {} pv {}",
                            mcts.get_depth(),
                            ((4.0 * ((mcts.get_root_q()) / (1.0 - mcts.get_root_q())).log10())
                                * 100.0) as i32,
                            i,
                            (i as f32 / now.elapsed().as_secs_f32()) as u32,
                            (mcts.total_size() as f32 / (40000.0 * 20.0) * 1000.0) as usize,
                            tbhits,
                            current_pv
                        );

                        last_pv = Some(current_pv);
                    }

                    if passed {
                        break;
                    }
                }
            });

            let san_pv = mcts
                .pv()
                .iter()
                .map(|m| San::from_move(&pos, m).to_string())
                .collect::<Vec<_>>()
                .join(", ");

            let mut distr = mcts.root_distribution(&pos);
            distr.sort_by(|b, a| a.1.partial_cmp(&b.1).unwrap());
            distr.truncate(5);
            println!("info string best line: {}", san_pv);
            for (mov, prob) in distr {
                println!(
                    "info string {}: {:.2}%",
                    San::from_move(&pos, &mov).to_string().to_string(),
                    prob * 100.0
                );
            }

            let pv = mcts.pv();
            println!(
                "bestmove {}",
                pv[0].to_uci(pos.castles().mode()).to_string()
            );
        });
    }

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
            }
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

                mcts.lock().unwrap().set_root(&pos, inference.clone());
            }
            UciMessage::Go { time_control, .. } => {
                tx.send((pos.clone(), time_control)).unwrap();
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            UciMessage::Stop => {
                should_stop.store(true, Ordering::Relaxed);
            }
            c => println!("error: {}", c),
        }
    }

    Ok(())

    /*let dev = AutoDevice::default();
    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();*/
}
