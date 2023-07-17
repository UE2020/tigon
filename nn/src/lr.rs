use burn::lr_scheduler::LRScheduler;
use burn::LearningRate;

#[derive(Clone, Debug)]
pub struct AlphaZeroLR {
    bounds: Vec<(LearningRate, usize)>,
    warmup_steps: usize,
    steps: usize,
}

impl AlphaZeroLR {
    pub fn new(lr: &[(LearningRate, usize)], warmup_steps: usize) -> Self {
        Self {
            bounds: lr.to_vec(),
            warmup_steps,
            steps: 0,
        }
    }
}

impl LRScheduler for AlphaZeroLR {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        let mut current_lr = 0.0;
        let mut last_checked = usize::MIN;
        for &(lr, step) in self.bounds.iter() {
            if self.steps < step && self.steps > last_checked {
                current_lr = lr;
                last_checked = step;
            }
        }
        if self.steps < self.warmup_steps {
            current_lr *= self.steps as LearningRate / self.warmup_steps as LearningRate;
        }
        self.steps += 1;
        current_lr
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
