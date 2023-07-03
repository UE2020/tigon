use std::sync::Mutex;

use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, NoStdTrainingRecorder, Recorder};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use burn::data::dataset::transform::PartialDataset;

use nn::*;
use pgn_reader::BufferedReader;

static ARTIFACT_DIR: &str = "/tmp/alphazero-checkpoints";

#[derive(Config)]
pub struct AlphaZeroTrainerConfig {
    #[config(default = 7)]
    pub num_epochs: usize,

    #[config(default = 1024)]
    pub batch_size: usize,

    //#[config(default = 4)]
    //pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub fn run<B: ADBackend>(device: B::Device) {
    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(None);
    let config = AlphaZeroTrainerConfig::new(config_optimizer);
    B::seed(config.seed);

    // Data
    let batcher_train = PositionBatcher::<B>::new(device.clone());
    let batcher_valid = PositionBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        //.num_workers(config.num_workers)
        .build(PositionDataset(Mutex::new(ChessPositionSet::new(
            BufferedReader::new(
                std::fs::File::open("nn/data/lichess_elite_2021-11.pgn")
                    .expect("training data not found"),
            ),
            10_000_000,
			1000000,
        ))));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        //.num_workers(config.num_workers)
        .build(PositionDataset(Mutex::new(ChessPositionSet::new(
            BufferedReader::new(
                std::fs::File::open("nn/data/lichess_elite_2021-11.pgn")
                    .expect("training data not found"),
            ),
            100000,
			100000,
        ))));

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train(AccuracyMetric::new())
        .metric_valid(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .with_file_checkpointer(1, NoStdTrainingRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(Model::new(5, 64), config.optimizer.init(), 1e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    NoStdTrainingRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{ARTIFACT_DIR}/model").into(),
        )
        .expect("Failed to save trained model");
}

use burn_autodiff::ADBackendDecorator;
use burn_tch::{TchBackend, TchDevice};

fn main() {
    #[cfg(not(target_os = "macos"))]
    let device = TchDevice::Cuda(0);
    #[cfg(target_os = "macos")]
    let device = TchDevice::Mps;

    run::<ADBackendDecorator<TchBackend<f32>>>(device);
}
