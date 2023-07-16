#![feature(iter_array_chunks)]

pub mod batch;
pub mod model;

pub use batch::*;
pub use model::*;

pub use burn;
pub use burn_tch;
