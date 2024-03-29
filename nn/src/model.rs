use burn::{
    module::Module,
    nn::{
        self,
        conv::Conv2dPaddingConfig,
        loss::{CrossEntropyLoss},
        BatchNorm,
    },
    tensor::{
        activation::sigmoid,
        backend::{ADBackend, Backend},
        Int, Tensor,
    },
    train::{
        metric::{AccuracyInput, Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

use crate::batch::PositionBatch;
use std::sync::Mutex;
use tensorboard_rs as tensorboard;

use once_cell::sync::OnceCell;

pub struct Writer(
    tensorboard::summary_writer::SummaryWriter,
    tensorboard::summary_writer::SummaryWriter,
    usize,
);

unsafe impl Sync for Writer {}

fn global_data() -> &'static Mutex<Writer> {
    static INSTANCE: OnceCell<Mutex<Writer>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        Mutex::new(Writer(
            tensorboard::summary_writer::SummaryWriter::new("logdir/train"),
            tensorboard::summary_writer::SummaryWriter::new("logdir/test"),
            0,
        ))
    })
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_block: ConvBlock<B>,
    residual_blocks: Vec<ResidualBlock<B>>,
    activation: nn::ReLU,
    policy_head: PolicyHead<B>,
    value_head: ValueHead<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(blocks: usize, filters: usize) -> Self {
        let input_block = ConvBlock::new(22, filters);
        Self {
            input_block,
            residual_blocks: vec![ResidualBlock::new(filters); blocks],
            activation: nn::ReLU::new(),
            policy_head: PolicyHead::new(filters),
            value_head: ValueHead::new(filters),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>, mask: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = self.input_block.forward(input);
        for block in self.residual_blocks.iter() {
            x = block.forward(x)
        }
        let policy = self.policy_head.forward(x.clone());
        let value = self.value_head.forward(x);
        (value, policy.mul(mask))
    }

    pub fn forward_output(&self, item: PositionBatch<B>) -> AlphaZeroOutput<B> {
        let (policy_targets, value_targets) = (item.policy_targets, item.value_targets);
        let (value_output, policy_output) = self.forward(item.positions, item.policy_masks);
        let cross_entropy = CrossEntropyLoss::new(None);
        let policy_loss = cross_entropy.forward(policy_output.clone(), policy_targets.clone());
        let value_loss = cross_entropy.forward(value_output.clone(), value_targets.clone());

        AlphaZeroOutput {
            policy_loss,
            value_loss,
            policy: policy_output,
            value: value_output,
            policy_targets,
			value_targets,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: nn::conv::Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    activation: nn::ReLU,

    se: SqueezeExcitationBlock<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(filters: usize) -> Self {
        let conv1 = nn::conv::Conv2dConfig::new([filters, filters], [3, 3])
            .with_padding(Conv2dPaddingConfig::Explicit(1, 1))
            .with_bias(false)
            .init();
        let bn1 = nn::BatchNormConfig::new(filters).init();

        let conv2 = nn::conv::Conv2dConfig::new([filters, filters], [3, 3])
            .with_padding(Conv2dPaddingConfig::Explicit(1, 1))
            .with_bias(false)
            .init();
        let bn2 = nn::BatchNormConfig::new(filters).init();

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            activation: nn::ReLU::new(),
            se: SqueezeExcitationBlock::new(filters, 2),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);

        let x = self.se.forward(x);

        let x = x.add(residual);
        let x = self.activation.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    activation: nn::ReLU,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(input_channels: usize, filters: usize) -> Self {
        let conv1 = nn::conv::Conv2dConfig::new([input_channels, filters], [3, 3])
            .with_padding(Conv2dPaddingConfig::Explicit(1, 1))
            .with_bias(false)
            .init();
        let bn1 = nn::BatchNormConfig::new(filters).init();

        Self {
            conv1,
            bn1,
            activation: nn::ReLU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);

        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    activation: nn::ReLU,
    fc1: nn::Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    pub fn new(filters: usize) -> Self {
        let conv1 = nn::conv::Conv2dConfig::new([filters, 8], [1, 1])
            .with_padding(Conv2dPaddingConfig::Valid)
            .with_bias(false)
            .init();
        let bn1 = nn::BatchNormConfig::new(8).init();
        let fc1 = nn::LinearConfig::new(8 * 8 * 8, 4608).init();

        Self {
            conv1,
            bn1,
            fc1,
            activation: nn::ReLU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let [batch_size, planes, height, width] = x.dims();
        let x = x.reshape([batch_size, planes * height * width]);
        let x = self.fc1.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,

    activation: nn::ReLU,
}

impl<B: Backend> ValueHead<B> {
    pub fn new(filters: usize) -> Self {
        let conv1 = nn::conv::Conv2dConfig::new([filters, 32], [1, 1])
            .with_padding(Conv2dPaddingConfig::Valid)
            .with_bias(false)
            .init();
        let bn1 = nn::BatchNormConfig::new(32).init();
        let fc1 = nn::LinearConfig::new(32 * 8 * 8, 256).init();
        let fc2 = nn::LinearConfig::new(256, 3).init();

        Self {
            conv1,
            bn1,
            fc1,
            fc2,
            activation: nn::ReLU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let [batch_size, planes, height, width] = x.dims();
        let x = x.reshape([batch_size, planes * height * width]);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct SqueezeExcitationBlock<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,

    activation: nn::ReLU,
}

impl<B: Backend> SqueezeExcitationBlock<B> {
    pub fn new(filters: usize, reduction_ratio: usize) -> Self {
        let fc1 = nn::LinearConfig::new(filters, filters / reduction_ratio)
            .with_bias(false)
            .init();
        let fc2 = nn::LinearConfig::new(filters / reduction_ratio, filters)
            .with_bias(false)
            .init();

        Self {
            fc1,
            fc2,
            activation: nn::ReLU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, planes, w, h] = input.dims();

        // global avg pool
        let x = input.clone().reshape([batch_size, planes, w * h]);
        let x = x.mean_dim(2);
        let x = x.reshape([batch_size, planes]);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = sigmoid(x);
        let x = x.reshape([batch_size, planes, 1, 1]);
        let x = x.mul(input);

        x
    }
}

pub struct AlphaZeroOutput<B: Backend> {
    pub policy_loss: Tensor<B, 1>,
    pub value_loss: Tensor<B, 1>,

    pub policy: Tensor<B, 2>,
    pub value: Tensor<B, 2>,

    pub policy_targets: Tensor<B, 1, Int>,
	pub value_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for AlphaZeroOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.policy.clone(), self.policy_targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for AlphaZeroOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.policy_loss.clone() + self.value_loss.clone())
    }
}

use num_traits::cast::ToPrimitive;

impl<B: ADBackend> TrainStep<PositionBatch<B>, AlphaZeroOutput<B>> for Model<B> {
    fn step(&self, item: PositionBatch<B>) -> TrainOutput<AlphaZeroOutput<B>> {
        let item = self.forward_output(item);
        let mut writer = global_data().lock().unwrap();
        let step = writer.2;
        writer.0.add_scalar(
            "Value Loss",
            (item.value_loss.clone()).into_data().value[0]
                .to_f32()
                .unwrap(),
            step,
        );
        writer.0.add_scalar(
            "Policy Loss",
            (item.policy_loss.clone()).into_data().value[0]
                .to_f32()
                .unwrap(),
            step,
        );
        writer.0.add_scalar(
            "Training loss",
            (item.policy_loss.clone()).into_data().value[0]
                .to_f32()
                .unwrap()
                + (item.value_loss.clone()).into_data().value[0]
                    .to_f32()
                    .unwrap(),
            step,
        );
        writer.2 += 1;

        TrainOutput::new(
            self,
            (item.value_loss.clone() + item.policy_loss.clone()).backward(),
            item,
        )
    }
}

impl<B: Backend> ValidStep<PositionBatch<B>, AlphaZeroOutput<B>> for Model<B> {
    fn step(&self, item: PositionBatch<B>) -> AlphaZeroOutput<B> {
        let out = self.forward_output(item);

        let mut writer = global_data().lock().unwrap();
        let step = writer.2;
        writer.1.add_scalar(
            "Value loss",
            (out.value_loss.clone()).into_data().value[0]
                .to_f32()
                .unwrap(),
            step,
        );
        writer.1.add_scalar(
            "Policy loss",
            (out.policy_loss.clone()).into_data().value[0]
                .to_f32()
                .unwrap(),
            step,
        );
        writer.1.add_scalar(
            "Training loss",
            (out.policy_loss.clone()).into_data().value[0]
                .to_f32()
                .unwrap()
                + (out.value_loss.clone()).into_data().value[0]
                    .to_f32()
                    .unwrap(),
            step,
        );
        writer.2 += 1;

        out
    }
}
