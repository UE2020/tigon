#![feature(generic_const_exprs)]

use shakmaty::fen::*;
use shakmaty::san::*;
use shakmaty::uci::*;
use shakmaty::variant::*;
use shakmaty::*;
use shakmaty_syzygy::{Syzygy, Tablebase, Wdl};

use std::io::{self, BufRead, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use colored::Colorize;
use nn::dfdx::{self, prelude::*, tensor::AutoDevice};
use vampirc_uci::{parse_one, UciMessage, UciTimeControl};

pub mod encoding;
pub mod mcts;

fn main() -> Result<(), PlayError<Chess>> {
    //tch::set_num_threads(4);
    eprintln!(
        "Current working directory: {}",
        std::env::current_dir().unwrap().display()
    );
    eprintln!("Loading neural network (64x5)");

    dfdx::flush_denormals_to_zero();

    let dev = AutoDevice::default();
    let mut model = dev.build_module::<nn::NetworkStructure<32, 4>, f32>();
    dbg!(model.num_trainable_params());
    model
        .load("/home/tt/Documents/tigon/testbed.npz")
        .expect("failed to load model");
    let model = Arc::new(model);
    let dev = Arc::new(dev);

    let mut history = mcts::HistoryTable::new();

    let mut log_file = std::fs::File::create("/tmp/tigon_log.txt").expect("cannot make log file");

    let tables = Arc::new(Mutex::new({
        let mut t = Tablebase::new();
        if std::path::Path::new("./tables").is_dir() {
            eprintln!("Automatically discovered Syzygy tables in ./tables");
            t.add_directory("./tables").expect("no tables");
        }
        t
    }));

    let inference = {
        let model = model.clone();
        let dev = dev.clone();
        Arc::new(move |pos: &Chess| {
            let data = encoding::encode_positions(pos);
            let tensor =
                dev.tensor_from_vec(data.into_raw_vec(), (Const::<16>, Const::<8>, Const::<8>));
            let (value_logits, policy_logits) = model.forward(tensor);
            let policy = (policy_logits
                * dev.tensor_from_vec(
                    encoding::legal_move_masks(pos).into_raw_vec(),
                    (Const::<4608>,),
                ))
            .softmax()
            .array();
            let value = value_logits.array()[0];
            let mut move_probabilities = Vec::new();
            let movegen = pos.legal_moves();
            for mov in movegen {
                if let Some(p) = mov.promotion() {
                    if p != Role::Queen {
                        continue;
                    }
                }
                let flip = pos.turn() == Color::Black;

                let (plane_idx, rank_idx, file_idx) = encoding::move_to_idx(&mov, flip);
                let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
                move_probabilities.push((mov, policy[mov_idx as usize]));
            }
            (
                value,
                move_probabilities
                    .into_iter()
                    .map(|(mov, prior)| {
                        let mut new_pos = pos.clone();
                        new_pos.play_unchecked(&mov);

                        let turn = new_pos.turn();

                        (
                            new_pos,
                            mov,
                            mcts::Node::new(
                                0.0,
                                prior,
                                mcts::turn_to_side(turn),
                                Some(mcts::default_hash(pos)),
                            ),
                            prior,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
    };

    let mut pos: Chess = Chess::default();
    let mcts = Arc::new(Mutex::new(mcts::MCTSTree::new(&pos, inference.clone())));
    let mut castling_mode = CastlingMode::Standard;
    let mut multipv = 1;

    let (tx, rx) = mpsc::channel();
    let should_stop = Arc::new(AtomicBool::new(false));
    {
        let should_stop = should_stop.clone();
        let mcts = mcts.clone();
        let inference = inference.clone();
        let tables = tables.clone();
        thread::spawn(move || 'outer: loop {
            let (pos, time_control, multipv, history): (
                Chess,
                Option<UciTimeControl>,
                u8,
                mcts::HistoryTable<mcts::FideHash>,
            ) = rx.recv().unwrap();
            should_stop.store(false, Ordering::Relaxed);

            let mut mcts = mcts.lock().unwrap();

            let mut target = match time_control {
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

                        let increment = white_increment
                            .unwrap_or(vampirc_uci::Duration::milliseconds(0))
                            .to_std()
                            .unwrap();

                        if pos.fullmoves().get() > 40 {
                            time_left / 20 + (increment / 2)
                        } else {
                            time_left / (40 - pos.fullmoves().get()) + (increment / 2)
                        }
                    }
                    _ => Duration::from_millis(3.6e+6 as u64),
                },
                None => Duration::from_millis(60000),
            };

            let now = Instant::now();
            let mut last_pv = None;
            let mut tbhits = 0;
            if (pos.board().black() | pos.board().white()).count()
                <= tables.lock().unwrap().max_pieces()
            {
                let best_move = tables.lock().unwrap().best_move(&pos);
                if let Ok(Some(mov)) = best_move {
                    println!(
                        "info depth 2 multipv 1 score mate {} pv {}",
                        -Wdl::from_dtz_after_zeroing(mov.1).signum(),
                        mov.0.to_uci(pos.castles().mode()).to_string(),
                    );
                    println!(
                        "info string {}: DTZ={}; WDL={}",
                        San::from_move(&pos, &mov.0).to_string(),
                        mov.1.ignore_rounding().0,
                        Wdl::from_dtz_after_zeroing(mov.1).signum()
                    );
                    println!(
                        "bestmove {}",
                        mov.0.to_uci(pos.castles().mode()).to_string()
                    );
                    continue;
                }
            }

            // look for mate in 1
            let legals = pos.legal_moves();
            for mov in legals {
                let mut new_pos = pos.clone();
                new_pos.play_unchecked(&mov);
                if let Some(Outcome::Decisive { .. }) = new_pos.outcome() {
                    // mate in 1 found
                    println!(
                        "info depth 2 multipv 1 score mate 1 pv {}",
                        mov.to_uci(pos.castles().mode()).to_string()
                    );
                    println!("bestmove {}", mov.to_uci(pos.castles().mode()).to_string());
                    continue 'outer;
                }
            }

            let mut was_extended = false;
            for i in 0.. {
                mcts.rollout(
                    Some(&tables.lock().unwrap()),
                    inference.clone(),
                    &mut tbhits,
                    history.clone(),
                )
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
                let has_elapsed = now.elapsed() >= target;
                if has_elapsed {
                    if mcts.is_critical() && !was_extended {
                        was_extended = true;
                        target += target / 2;
                    }
                }
                let passed =
                    now.elapsed() >= target || (now.elapsed() >= (target / 2) && mcts.is_easy());
                if last_pv.as_ref() != Some(&current_pv) || passed || i % 150 == 0 {
                    let mut all_pvs = mcts.all_pvs();
                    all_pvs.sort_by(|b, a| a.0.cmp(&b.0));
                    all_pvs.truncate(multipv as usize);
                    for (multipv, (_, score, pv)) in all_pvs.into_iter().enumerate() {
                        println!(
							"info depth {} multipv {} score cp {} nodes {} nps {} hashfull {} tbhits {} pv {}",
							mcts.get_depth(),
							multipv + 1,
							mcts::q_to_cp(score),
							i,
							(i as f32 / now.elapsed().as_secs_f32()) as u32,
							(mcts.total_size() as f32 / (40000.0 * 20.0) * 1000.0) as usize,
							tbhits,
							pv.iter()
							.map(|m| m.to_uci(pos.castles().mode()).to_string())
							.collect::<Vec<_>>()
							.join(" ")
						);
                    }

                    last_pv = Some(current_pv);
                }

                if passed {
                    break;
                }
            }

            let san_pv = mcts
                .pv()
                .iter()
                .map(|m| San::from_move(&pos, m).to_string())
                .collect::<Vec<_>>()
                .join(" ");

            let mut distr = mcts.root_distribution();
            distr.sort_by(|b, a| a.1.partial_cmp(&b.1).unwrap());
            distr.truncate(10);
            eprintln!(
                "Calculated best line: {}\nProbability distribution:",
                san_pv.italic()
            );
            for (i, (mov, prob)) in distr.into_iter().enumerate() {
                eprintln!(
                    "{}: {}%",
                    San::from_move(&pos, &mov).to_string().to_string().bold(),
                    {
                        let s = format!("{:.2}", (prob * 100.0));
                        match i {
                            0 => s.green(),
                            1..=3 => s.yellow(),
                            _ => s.red(),
                        }
                    }
                );
            }

            let pv = mcts.pv();
            println!(
                "bestmove {}",
                pv[0].to_uci(pos.castles().mode()).to_string()
            );
        });
    }

    eprintln!("{}", include_str!("banner.txt"));

    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        writeln!(log_file, "{}", line).ok();
        let msg: UciMessage = parse_one(&line);
        match msg {
            UciMessage::Uci => {
                println!("id name TigonNN");
                println!("id author Aspect8445");
                println!("option name UCI_Chess960 type check default false");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name MultiPV type spin default 1 min 1 max 255");
                println!("uciok")
            }
            UciMessage::UciNewGame => {
                mcts.lock().unwrap().clear(inference.clone());
                history.clear();
            }
            UciMessage::SetOption { name, value } => {
                if name == "UCI_Chess960" {
                    if &value.unwrap() == "true" {
                        castling_mode = CastlingMode::Chess960;
                        eprintln!("Using chess960");
                    } else {
                        castling_mode = CastlingMode::Standard;
                    }
                } else if name == "SyzygyPath" {
                    match value {
                        Some(value) => {
                            if &value == "<empty>" {
                                *tables.lock().unwrap() = Tablebase::new();
                            }
                            tables
                                .lock()
                                .unwrap()
                                .add_directory(value)
                                .expect("tables not found");
                        }
                        None => {
                            *tables.lock().unwrap() = Tablebase::new();
                        }
                    }
                } else if name == "MultiPV" {
                    if let Some(value) = value {
                        multipv = value.parse().expect("invalid multipv");
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

                history.clear();
                history
                    .entry(mcts::fide_hash(pos.clone()))
                    .and_modify(|counter| *counter += 1)
                    .or_insert(1);

                for mov in moves {
                    let uci = mov.to_string();
                    let uci: Uci = uci.parse().expect("bad uci");
                    let m = uci.to_move(&pos).expect("bad move");
                    pos.play_unchecked(&m);
                    history
                        .entry(mcts::fide_hash(pos.clone()))
                        .and_modify(|counter| *counter += 1)
                        .or_insert(1);
                }

                mcts.lock().unwrap().set_root(&pos, inference.clone());
            }
            UciMessage::Go { time_control, .. } => {
                tx.send((pos.clone(), time_control, multipv, history.clone()))
                    .unwrap();
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => std::process::exit(0),
            UciMessage::Stop => {
                should_stop.store(true, Ordering::Relaxed);
            }
            c => println!("[INTERNAL ERROR] {}", c),
        }
    }

    Ok(())

    /*let dev = AutoDevice::default();
    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();*/
}
