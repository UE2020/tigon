use burn::lr_scheduler::LRScheduler;
use burn::LearningRate;

#[derive(Clone, Debug)]
pub struct AlphaZeroLR {
    lr: LearningRate,
    step_size: usize,
    steps: usize,
}

impl AlphaZeroLR {
    pub fn new(lr: LearningRate, step_size: usize) -> Self {
        Self {
            lr,
            step_size,
            steps: 0,
        }
    }
}

impl LRScheduler for AlphaZeroLR {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        self.steps += 1;
        if self.steps % self.step_size == 0 {
            self.lr /= 10.0;
        }
        self.lr
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
