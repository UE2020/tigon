use burn::lr_scheduler::LRScheduler;
use burn::LearningRate;
use once_cell::sync::OnceCell;
use tensorboard_rs as tensorboard;
use std::sync::Mutex;

pub struct Writer(tensorboard::summary_writer::SummaryWriter);

unsafe impl Sync for Writer {}

fn global_data() -> &'static Mutex<Writer> {
    static INSTANCE: OnceCell<Mutex<Writer>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        Mutex::new(Writer(
            tensorboard::summary_writer::SummaryWriter::new("logdir/general")
        ))
    })
}

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
		global_data().lock().unwrap().0.add_scalar("Learning Rate", current_lr as f32, self.steps);
        current_lr
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
