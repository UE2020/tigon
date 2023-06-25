#![feature(generic_const_exprs)]

use std::fs::File;
use std::time::Instant;

use dfdx::optim::{AdamConfig, Momentum, Sgd, SgdConfig, WeightDecay};
use dfdx::{data::*, optim::Adam, prelude::*, tensor::AutoDevice};
use indicatif::ProgressIterator;
use pgn_reader::*;
use rand::prelude::{SeedableRng, StdRng};
use shakmaty::*;

use tensorboard_rs as tensorboard;

use nn::*;

const BATCH_SIZE: usize = 1024;

fn main() {
    dfdx::flush_denormals_to_zero();

    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(0);

    let args: Vec<String> = std::env::args().collect();

    let mut model = dev.build_module::<NetworkStructure<64, 5>, f32>();
    // model.load("testbed.npz").expect("failed to load model");

    // let pos: Chess = shakmaty::fen::Fen::from_ascii(args[1].as_bytes())
    //     .expect("bad fen")
    //     .into_position(CastlingMode::Standard)
    //     .expect("bad fen");
    // dbg!(&args[1]);
    // let data = data::encode_positions(&pos);
    // let tensor = dev.tensor_from_vec(data.into_raw_vec(), (Const::<16>, Const::<8>, Const::<8>));
    // let (value_logits, policy_logits) = model.forward(tensor);
    // let policy = (policy_logits
    //     * dev.tensor_from_vec(
    //         data::legal_move_masks(&pos).into_raw_vec(),
    //         (Const::<4608>,),
    //     ))
    // .softmax();
    // println!(
    //     "Q value: {:.2}%",
    //     (value_logits.array()[0] / 2.0 + 0.5) * 100.0
    // );
    // let mut move_probabilities = Vec::new();
    // let movegen = pos.legal_moves();
    // for mov in movegen {
    //     if let Some(p) = mov.promotion() {
    //         if p != Role::Queen {
    //             continue;
    //         }
    //     }
    //     let flip = pos.turn() == Color::Black;

    //     let (plane_idx, rank_idx, file_idx) = data::move_to_idx(&mov, flip);
    //     let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
    //     move_probabilities.push((mov, policy[[mov_idx as usize]]));
    // }

    // for (mov, prob) in move_probabilities {
    //     println!(
    //         "Move {}: {:.2}%",
    //         San::from_move(&pos, &mov).to_string(),
    //         prob * 100.0
    //     );
    // }

    // return;

    let mut grads = model.alloc_grads();
    let mut opt = Adam::new(
        &model,
        AdamConfig {
            // lr: 0.01,
            // momentum: Some(Momentum::Nesterov(0.9)),
            // weight_decay: Some(WeightDecay::L2(1e-4)),
            ..Default::default()
        },
    );

    let preprocess =
        |(img, eval, policy, mask): <data::ChessPositionSet as ExactSizeDataset>::Item<'_>| {
            (
                dev.tensor_from_vec(img, (Const::<16>, Const::<8>, Const::<8>)),
                (
                    dev.tensor([eval]),
                    (
                        dev.tensor_from_vec(policy, (Const::<4608>,)),
                        dev.tensor_from_vec(mask, (Const::<4608>,)),
                    ),
                ),
            )
        };

    let mut writer = tensorboard::summary_writer::SummaryWriter::new("logdir");

    let mut total_training_steps = 0;

    for i_epoch in 0..7 {
        let file = File::open("nn/data/lichess_elite_2021-10.pgn")
            .expect("training data not found");
        let mut reader = BufferedReader::new(file);
        let mut visitor = data::PgnVisitor::new();

        loop {
            let mut should_stop = false;
            let mut total_epoch_loss = 0.0;
            let mut num_batches = 0;
            let start = Instant::now();
            let mut games = vec![];
            for _ in 0..250000 {
                if let Some(Some(result)) =
                    reader.read_game(&mut visitor).expect("failed to read game")
                {
                    games.push(result)
                } else {
                    should_stop = true;
                    break;
                }
            }

            let dataset = data::ChessPositionSet::new(games);

            for (img, labels) in dataset
                .shuffled(&mut rng)
                .map(preprocess)
                .batch_exact(Const::<BATCH_SIZE>)
                .collate()
                //.stack()
                .progress()
            {
                total_training_steps += 1;
                if total_training_steps % 20000 == 0 {
                    opt.cfg.lr /= 10.0;
                }

                if total_training_steps % 20 == 0 {
                    model.save("testbed.npz").expect("failed to save model");
                    //println!("Saved model at {} steps", total_training_steps);
                }
                // stack data
                let data = img.stack();
                let logits = model.forward_mut(data.traced(grads));
                let (evals, policies) = labels.collated();
                let (policies, masks) = policies.collated();
                let targets = policies.stack();

                let value = mse_loss(logits.0, evals.stack());
                let policy = cross_entropy_with_logits_loss(logits.1 * masks.stack(), targets);
                writer.add_scalar("Value loss", value.array(), total_training_steps as usize);
                writer.add_scalar("Policy loss", policy.array(), total_training_steps as usize);
                let loss = value + policy;

                writer.add_scalar("Training loss", loss.array(), total_training_steps as usize);

                total_epoch_loss += loss.array();
                num_batches += 1;

                grads = loss.backward();
                opt.update(&mut model, &grads).unwrap();
                model.zero_grads(&mut grads);
            }
            let dur = Instant::now() - start;

            println!(
                "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
                dur,
                num_batches as f32 / dur.as_secs_f32(),
                BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
            );

            model.save("testbed.npz").expect("failed to save model");
            if should_stop {
                break;
            }
        }
    }
}
