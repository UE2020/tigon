use dfdx::prelude::*;
pub mod mcts;

type Mlp = (
    (Linear<5, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);


fn main() {
	let dev = AutoDevice::default();
    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();

	
}
