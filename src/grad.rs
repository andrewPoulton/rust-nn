use crate::linalg::*;
use crate::errors::TensorError;
pub trait Module {
    fn forward(&mut self, input: &Tensor)->Result<Tensor, TensorError>;
    fn backward(&self, grad_input: &Tensor)->Result<Tensor, TensorError>;
}