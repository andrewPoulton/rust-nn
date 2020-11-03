use crate::errors::*;
use crate::linear_layer::*;
use crate::activations::*;
use crate::linalg::*;
use crate::grad::Module;

pub struct FFNet {
    pub layers: Vec<Linear>,
    pub activations: Vec<Activation>,
}

impl FFNet {
    pub fn new(dims: &[usize], hidden_nonlinearity: NonLinearity)->FFNet{
        let mut layers = vec![Linear::new(2, dims[0])];
        let mut activations: Vec<Activation> = Vec::<Activation>::with_capacity(dims[1..].len());
        for i in 1..dims.len() {
            layers.push(Linear::new(dims[i-1], dims[i]));
            activations.push(Activation::new(hidden_nonlinearity));
        }
        activations.push(Activation::new(NonLinearity::Sigmoid));
        FFNet { layers, activations }
    }

    pub fn init_weight(&mut self, mean: f32, std: f32){
        self.layers.iter_mut().map(|layer| layer.init_weight(mean, std));
    }
}

impl Module for FFNet {

    fn forward(&mut self, input: &Tensor)->Result<Tensor, TensorError> {
        let mut hidden_state = input.clone();
        for (i, layer) in self.layers.iter_mut().enumerate(){
            hidden_state = layer.forward(&hidden_state).unwrap();
            hidden_state = self.activations[i].forward(&hidden_state).unwrap();
        };
        Ok(hidden_state)
    }

    fn backward(&self, grad_input: &Tensor)->Result<Tensor, TensorError> {
        todo!()
    }
}