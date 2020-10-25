pub mod linalg;
pub mod errors;
pub mod linear_layer;
pub mod activations;
pub mod loss;
pub mod grad;
pub mod nn;
pub use activations::*;
pub use linalg::*;
pub use linear_layer::*;
pub use loss::*;
pub use grad::Module;
pub use nn::FFNet;

pub fn mmul_()->Tensor{
    let v1 = Tensor::uniform(&[5,10], 0.0, 1.0).unwrap();
    let v2 = Tensor::uniform(&[10, 5], 0.0, 1.0).unwrap();
    v2.mmul(&v1).unwrap()
}

pub fn linlayer()->Tensor{
    let v = Tensor::uniform(&[10, 5], 0.0, 1.0).unwrap();
    let mut l = Linear::new(5, 1);
    l.forward(&v).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    // use linalg::Tensor;
    
    
    #[test]
    fn not_run(){
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4, 1]).unwrap();
        // let v = Vector::new(data);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn norm_test(){
        let v = Tensor::new(vec![3.0,4.0], &[2]).unwrap();
        println!("{}", v);
        assert_eq!(v.norm(), 5.0);
    }
    
    #[test]
    fn vec_length_test(){
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4, 1]).unwrap();
        // let v = Vector::new(data);
        assert_eq!(t.numel(), 4);
    }
    #[test]
    fn slice_test(){
        let v = Tensor::new((0u8..10).map(f32::from).collect(), &[2,5]).unwrap();
        println!("{:?}", &v[2..6]);
    }  


    #[test]
    fn row_col_idx_test(){
        let v = Tensor::new((1u8..13).map(f32::from).collect(), &[3,4]).unwrap();
        let idx = v.idx_from_row_column(2, 1);
        assert_eq!(10.0, v[idx]);
    }

    #[test]
    fn mmul_test(){
        let v1 = Tensor::new(vec![1.0, 1.0, 1.0], &[3,1]).unwrap();
        let v2 = Tensor::new((1u8..13).map(f32::from).collect(), &[4,3]).unwrap();
        let v3 = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], &[4,1]).unwrap();
        let v4 = v2.mmul(&v1);
        println!("mult test: {}", v4.clone().unwrap());
        println!("col add test: {}", v4.unwrap().add(&v3).unwrap());
    }

    #[test]
    fn add_test(){
        let v1 = Tensor::new((0u8..12).map(f32::from).collect(), &[4,3]).unwrap();
        println!("{}", v1.add(&v1).unwrap());
        let v2 = Tensor::new((1u8..5).map(f32::from).collect(), &[4,1]).unwrap();
        let v3 = Tensor::new((1u8..4).map(f32::from).collect(), &[1,3]).unwrap();
        println!("v2 is {}", v2);
        println!("v1 is {}", v1);
        println!("v1 + v2 = {}",v1.add(&v2).unwrap());
        println!("v3 is {}", v3);
        println!("v1 + v3 = {}",v1.add(&v3).unwrap());
    }

    #[test]
    fn scalar_mult_test(){
        let mut v1 = Tensor::new((0u8..12).map(f32::from).collect(), &[4,3]).unwrap();
        v1 *= 3.0;
        println!("v1*12 = {}", 2.0*v1*2.0);
    }

    #[test]
    fn linear_test(){
        let mut l = Linear::new(4, 5);
        let mut new_weight = Tensor::new((0u8..20).map(f32::from).collect(), &[4,5]);
        let t = Tensor::uniform(&[10, 2], -1.0, 1.0).unwrap();
        println!("l out is {}", l.forward(&t).unwrap());
        l.init_weight(0.0, 2.0);
        println!("l newly init out is {}", l.forward(&t).unwrap());
    }

    #[test]
    fn mse_test(){
        let targets = Tensor::ones(&[5]);
        let preds = Tensor::from_vec(vec![0.0,1.0,2.0,3.0,4.0]).unwrap();
        let l = mse_loss(&preds, &targets).unwrap();
        println!("\nloss is {}\n", l.loss);
        println!("\nloss grad is {}\n", l.loss_grad)
    }

    #[test]
    fn activation_backwards_test(){
        let preds = Tensor::from_vec(vec![0.0,-1.0,2.0,-3.0,4.0]).unwrap();
        let mut act = Activation::from_tensor(&preds,NonLinearity::ReLU);
        let back_grads = Tensor::from_vec(vec![5.0, 2.0, 1.4, 6.8, 9.3]).unwrap();
        let grad = act.backward(&back_grads);
        println!("backwards result is {}", grad.unwrap())
    }

    #[test]
    fn transpose_test(){
        let mut preds = Tensor::from_vec(vec![0.0,-1.0,2.0,-3.0, 4.0, 5.0]).unwrap();
        preds.shape = vec![3,2];
        println!("Tensor untransposed is {}", preds);
        let preds = preds.transpose();
        println!("Tensor transposed is {}", preds);
        let preds = preds.transpose();
        println!("Tensor transposed transposed is {}", preds);
    }

    #[test]
    fn ff_test(){
        
    }

}


