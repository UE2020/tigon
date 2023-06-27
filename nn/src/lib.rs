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
    ((Conv2D<22, FILTERS, 3, 1, 1>, BatchNorm2D<FILTERS>), ReLU),
    Repeated<(BasicBlock<FILTERS>, ReLU), BLOCKS>,
    SplitInto<(ValueHead<FILTERS>, PolicyHead<FILTERS>)>,
);

pub type DenseNetworkStructure<const FILTERS: usize, const SIZE: usize, const BLOCKS: usize> = (
    ((Conv2D<22, FILTERS, 3, 1, 1>, BatchNorm2D<FILTERS>), ReLU),
    (Flatten2D, Linear<4096, SIZE>, BatchNorm1D<SIZE>),
    Repeated<DenseBlock<SIZE>, BLOCKS>,
    (BatchNorm1D<SIZE>, ReLU),
    SplitInto<(
        (
            Linear<SIZE, 64>,
            Linear<64, 256>,
            ReLU,
            Linear<256, 1>,
            Tanh,
        ),
        (Linear<SIZE, 4608>,),
    )>,
);

pub type DenseBlock<const SIZE: usize> = Residual<(
    BatchNorm1D<SIZE>,
    ReLU,
    Linear<SIZE, SIZE>,
    BatchNorm1D<SIZE>,
    ReLU,
    Linear<SIZE, SIZE>,
)>;

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
