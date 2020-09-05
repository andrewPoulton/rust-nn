use crate::{linalg::*, errors::TensorError};
use crate::activations::*;
use crate::grad::Module;




pub struct Linear{
    pub weight: Tensor,
    pub bias: Tensor,
    pub weight_grad: Tensor,
    pub bias_grad: Tensor,
    pub output: Option<Tensor>,
    pub input: Option<Tensor>,
}

impl Linear{
    pub fn new(in_dim: usize, out_dim: usize)->Self{
        let bias = Tensor::zeros(&[out_dim, 1]);
        let weight = Tensor::normal(&[out_dim, in_dim], 0.0, 0.01).unwrap();
        let bias_grad = Tensor::zeros_like(&bias);
        let weight_grad = Tensor::zeros_like(&weight);
        Linear{
            weight: weight.transpose(),
            bias,
            weight_grad,
            bias_grad,
            output: None, 
            input: None
        }
    }

    pub fn init_weight(&mut self, mean: f32, std: f32){
        self.weight = match Tensor::normal(&self.weight.shape[..], mean, std) {
            Ok(a) => a,
            _ => unreachable!(),
        };
    }

    pub fn set_weight(&mut self, weight: &Tensor)->Result<(), TensorError>{
        if self.weight.shape == weight.shape  {
            self.weight = weight.clone();
            Ok(())
        } else {
            Err(TensorError::ShapeError)
        }
    }
}

impl Module for Linear{

    fn forward(&mut self, input: &Tensor)->Result<Tensor, TensorError> {
        self.input = Some(input.to_owned());
        if input.shape[1] == self.weight.shape[0]{
            let logits = input.mmul(&self.weight).unwrap().add(&self.bias);
            self.output = Some(logits.to_owned().unwrap());
            logits
        } else {
            Err(TensorError::MatmulShapeError)
        }
    }

    fn backward(&self, grad_input: &Tensor)->Result<Tensor, TensorError> {
        todo!()
    }
}