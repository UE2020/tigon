pub use dfdx;
use dfdx::prelude::*;

pub mod data;
pub mod visuals;

pub type BasicBlock<const C: usize> = Residual<(
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
    ReLU,
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
)>;

pub type NetworkStructure<const FILTERS: usize, const BLOCKS: usize> = (
    ((Conv2D<16, FILTERS, 3, 1, 1>, BatchNorm2D<FILTERS>), ReLU),
    visuals::PrintOutput<Repeated<(BasicBlock<FILTERS>, ReLU), BLOCKS>>,
    SplitInto<(ValueHead<FILTERS>, PolicyHead<FILTERS>)>,
);

pub type ValueHead<const FILTERS: usize> = (
    ((Conv2D<FILTERS, 1, 1>, BatchNorm2D<1>), ReLU),
    (Flatten2D, Linear<64, 256>, ReLU, Linear<256, 1>, Tanh),
);

pub type PolicyHead<const FILTERS: usize> = (
    ((Conv2D<FILTERS, 2, 1>, BatchNorm2D<2>), ReLU),
    (Flatten2D, Linear<128, 4608>),
);

pub type Model<const FILTERS: usize, const BLOCKS: usize> =
    <NetworkStructure<FILTERS, BLOCKS> as BuildOnDevice<AutoDevice, f64>>::Built;
