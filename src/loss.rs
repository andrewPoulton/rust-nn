use crate::linalg::*;
use crate::errors::*;
use crate::grad::Module;

#[derive(Debug, Clone)]
pub struct Loss {
    pub loss: f32,
    pub loss_grad: Tensor
}

pub fn mse_loss(predictions:&Tensor, target:&Tensor)->Result<Loss, TensorError>{
    match (predictions.tensor_type, target.tensor_type) {
        (TensorType::Matrix, TensorType::Matrix) => {
            Err(TensorError::ShapeError)
        },
        _ => {
            let batch_size = predictions.data.len() as f32;
            let loss = predictions.data.iter()
                                        .zip(target.data.iter())
                                        .map(|(x,y)| (x-y).powi(2))
                                        .sum::<f32>();
            let grad: Vec<f32> = predictions.data.iter()
            .zip(target.data.iter())
            .map(|(x,y)| 2.0*(x-y)/batch_size)
            .collect();
            
            Ok(Loss { loss: loss/ batch_size, loss_grad : Tensor::from_vec(grad).unwrap() })
        }
    }
}
#[inline]
fn log_and_clamp(x: f32, min: f32, max: f32)->f32{
    let out = x.ln();
    if out > max {max} else if out < min {min} else {out}
}

pub fn bce_loss(predictions:&Tensor, target:&Tensor)-> Result<Loss, TensorError>{
    match (predictions.tensor_type, target.tensor_type) {
        (TensorType::Matrix, TensorType::Matrix) => {
            Err(TensorError::ShapeError)
        },
        _ => {
            let batch_size = predictions.data.len() as f32;
            let loss = predictions.data.iter()
                .zip(target.data.iter())
                .map(|(x,y)| y*log_and_clamp(*x,-100.0f32, 100.0f32) + (1.0-y)*log_and_clamp(1.0-x,-100.0, 100.0))
                .sum::<f32>();
            let grad: Vec<f32> = predictions.data.iter()
                .zip(target.data.iter())
                .map(|(x,y)| (x-y)/(batch_size*x*(1.0-x)))
                .collect();
            
            Ok(Loss { loss: -loss/ batch_size, loss_grad : Tensor::from_vec(grad).unwrap() })
        }
    }
}