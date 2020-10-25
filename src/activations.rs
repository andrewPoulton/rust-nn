use crate::linalg::*;
use crate::errors::TensorError;
use crate::grad::Module;

#[derive(Debug, Copy, Clone)]
pub enum NonLinearity{
    ReLU,
    Tanh, 
    Sigmoid
}
#[derive(Debug,Clone)]
pub struct Activation{
    // pub input: Option<Tensor>,
    pub output: Option<Tensor>,
    pub func: NonLinearity
}

#[inline]
fn sigmoid(x:&f32)-> f32{
    (1.0 + (-*x).exp()).recip()
}

#[inline]
fn sigmoid_backward(x: &f32)->f32{
    sigmoid(x)*(1.0-sigmoid(x))
}

#[inline]
fn relu(x:&f32)->f32{
    if *x > 0.0 { *x } else {0.0}
}

#[inline]
fn relu_backward(x:&f32)->f32{
    if *x > 0.0 { 1.0 } else {0.0}
}

#[inline]
fn tanh_backward(x:&f32)->f32{
    1.0 - x.tanh().powi(2)
}

impl Activation{
    pub fn new(func:NonLinearity)-> Self{
        Activation {
            // input:None, 
                    output:None, 
                    func
                }
    }

    pub fn from_tensor(tensor: &Tensor, func:NonLinearity)->Self{
        let mut act = Activation::new(func);
        act.forward(tensor);
        act
    }    
}

impl Module for Activation {

    fn forward(&mut self, input: &Tensor)->Result<Tensor, TensorError>{
        self.output = match self.func {
            NonLinearity::ReLU => {
                let out_data = input.data.iter().map(|elem| relu(elem) ).collect();
                Some(Tensor::from_vec(out_data).unwrap())
            },
            NonLinearity::Tanh => {
                let out_data = input.data.iter().map(|elem| elem.tanh() ).collect();
                Some(Tensor::from_vec(out_data).unwrap())
            },
            NonLinearity::Sigmoid => {
                let out_data = input.data.iter().map(|elem| sigmoid(elem)).collect();
                Some(Tensor::from_vec(out_data).unwrap())
            }
        };
        Ok(self.output.to_owned().unwrap())
    }

    fn backward(&self, grad_input: &Tensor)-> Result<Tensor, TensorError>{

        match &self.output {
            None => Err(TensorError::NoForwardError),
            Some(activations) => {
                if grad_input.shape != activations.shape {
                    Err(TensorError::GradError)
                } else {
                    match self.func {
                        NonLinearity::ReLU => {
                            let grad_data = activations.data
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(idx, x)| relu_backward(x)*grad_input.data[idx])
                                                                .collect();
                            
                            Ok(Tensor::from_vec(grad_data)?)
                        },
                        NonLinearity::Sigmoid => {
                            let grad_data = activations.data
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(idx, x)| sigmoid_backward(x)*grad_input.data[idx])
                                                                .collect();
                            Ok(Tensor::from_vec(grad_data)?)
                        }
                        NonLinearity::Tanh => {
                            let grad_data = activations.data
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(idx, x)| tanh_backward(x)*grad_input.data[idx])
                                                                .collect();
                            Ok(Tensor::from_vec(grad_data)?)
                        }
                    }
                }
            }
        }
    }
}
