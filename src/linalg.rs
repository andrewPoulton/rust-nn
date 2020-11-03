use std::ops::{Mul, MulAssign, AddAssign};
use crate::errors::TensorError;

use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use std::fmt;
#[derive(Clone, Debug, Copy)]
pub enum TensorType{
    RowVector,
    ColumnVector,
    Matrix,
}
#[derive(Clone, Debug)]
pub struct Tensor{
    pub data: Vec<f32>,
    pub shape:  Vec<usize>,
    pub strides: Vec<usize>,
    pub tensor_type: TensorType,
}




impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!(
            "\nValue: {:?} \nShape: {:?} \nType: {:?}",
            self.data, self.shape, self.tensor_type))
    }
}

impl Tensor{
    fn calc_tensor_len_from_shape(shape: &[usize]) -> usize {
        let mut length = 1;
        for i in shape {
            length *= i;
        }

        length
    }

    fn calc_strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());

        let mut current_stride = 1;
        for i in shape.iter().rev() {
            strides.insert(0, current_stride);
            current_stride *= i;
        }

        strides
    }

    pub fn new(data: Vec<f32>, shape: &[usize]) -> Result<Tensor, TensorError> {
        if data.len() == Tensor::calc_tensor_len_from_shape(shape)
            && !shape.is_empty()
            && shape.len() < 3 // only allow upto rank-2 tensors, identify column and row vectors, default to row vectors 
        {   
            let tensor_type: Result<TensorType, TensorError> = match shape.len() {
                1 => Ok(TensorType::ColumnVector),
                2 => {
                    if shape[0] == 1 {
                        Ok(TensorType::RowVector)
                    } else if shape[1] == 1{
                        Ok(TensorType::ColumnVector)
                    } else {
                        Ok(TensorType::Matrix)
                    }
                }
                _ => Err(TensorError::InvalidTensor)
            };
            let mut final_shape = shape.to_vec();
            if final_shape.len() == 1 {
                final_shape.push(1);
            }

            Ok(Tensor {
                data,
                shape: final_shape,
                strides: Tensor::calc_strides_from_shape(shape),
                tensor_type: tensor_type.unwrap(),
            })
        } else {
            Err(TensorError::InvalidTensor)
        }
    }

    pub fn from_vec(vec: Vec<f32>)-> Result<Tensor, TensorError>{
        let shape = &[(&vec).len()];
        Tensor::new(vec, shape)
    }

    pub fn dims(&self)->usize{
        self.shape.len()
    }

    pub fn numel(&self)->usize{
        self.data.len()
    }

    pub fn transpose(&self)->Tensor{
        let mut out_tensor = Tensor::zeros_like(self);
        out_tensor.shape = vec![self.shape[1], self.shape[0]];
        out_tensor.tensor_type = match self.tensor_type {
            TensorType::Matrix => TensorType::Matrix,
            TensorType::ColumnVector => TensorType::RowVector,
            TensorType::RowVector => TensorType::ColumnVector
        };
        for (idx, elem) in self.data.iter().enumerate(){
            let (row, col) = (idx/self.shape[1], idx % self.shape[1]);
            let c_idx = out_tensor.idx_from_row_column(col, row);
            out_tensor.data[c_idx] = *elem
        }
        out_tensor
    }

    pub fn idx_from_row_column(&self, row:usize, col:usize)->usize{
        self.shape[1]*row + col
    }

    pub fn elt_from_row_column(&self, row:usize, col:usize)->f32{
        self[self.idx_from_row_column(row, col)]
    }

   
    pub fn mmul(&self, other: &Tensor) -> Result<Tensor, TensorError>{
        if Some(&other.shape[0]) == self.shape.last() {
            let mut c: Vec<f32> = vec![0.0; Tensor::calc_tensor_len_from_shape(&[self.shape[0], other.shape[1]])];
            let c_shape = &[self.shape[0], other.shape[1]];
            
            for i in 0..self.shape[0]{
                let start_idx = i*self.shape[1];
                let end_idx = start_idx+self.shape[1];
                let row = &self[start_idx..end_idx];
                for j in 0..other.shape[1]{
                    let c_idx = c_shape[1]*i + j;
                    for (idx, k) in row.iter().enumerate(){
                        c[c_idx] += k*other.elt_from_row_column(idx, j)
                    }
                }
            }
            Ok(Tensor::new(c, c_shape)?)
        } else {
            Err(TensorError::MatmulShapeError)
        }
    }

    fn add_(&self, other: &Tensor)-> Result<Tensor, TensorError>{
        if self.shape == other.shape{
            let new_data = self.data.iter().zip(other.data.iter()).map(|(x,y)| x+y).collect();
            Tensor::new(new_data, &self.shape[0..])
        } else{
            Err(TensorError::AddError)
        }
    }

    fn add_vector(&self, other: &Tensor)->Result<Tensor, TensorError>{
        match other.tensor_type {
            TensorType::Matrix => Err(TensorError::MaxDimsError),
            TensorType::RowVector => {
                let other_length = self.shape[1];
                if other_length == other.shape[1]{
                    let new_data = self.data
                                            .clone()
                                            .iter()
                                            .enumerate()
                                            .map(|(idx,x)|x+other[idx%other_length])
                                            .collect();
                    Tensor::new(new_data, &self.shape[..])
                } else {
                    Err(TensorError::MatmulShapeError)
                }
            },
            TensorType::ColumnVector => {
                if self.shape[0] == other.shape[0]{
                    let new_data = self.data
                                            .clone()
                                            .iter()
                                            .enumerate()
                                            .map(|(idx,x)|x+other[idx/self.shape[1]])
                                            .collect();
                    Tensor::new(new_data, &self.shape[..])
                } else {
                    Err(TensorError::OpError)
                }
            }
        }
    }

    pub fn add(&self, rhs: &Tensor)-> Result<Tensor, TensorError>{
        match self.tensor_type {
            TensorType::Matrix => match rhs.tensor_type {
                TensorType::Matrix => self.add_(rhs),
                TensorType::RowVector => self.add_vector(rhs),
                TensorType::ColumnVector => self.add_vector(rhs)
            },
            TensorType::ColumnVector => match rhs.tensor_type {
                TensorType::Matrix => rhs.add_vector(self),
                TensorType::RowVector => self.add_vector(rhs),
                TensorType::ColumnVector => self.add_(rhs)
            },
            TensorType::RowVector => match rhs.tensor_type {
                TensorType::Matrix => rhs.add_vector(self),
                TensorType::RowVector => self.add_(rhs),
                TensorType::ColumnVector => self.add_(rhs)
            }

        }
    }


    pub fn norm(&self)->f32{
        self.data.iter().map(|x| x*x).sum::<f32>().sqrt()
    }

    pub fn zeros_like(tensor: &Tensor)->Tensor{
        let data = vec![0.0;  Tensor::calc_tensor_len_from_shape(&tensor.shape[..])];
        Tensor::new(data, &tensor.shape[..]).unwrap()
    }

    pub fn zeros(shape: &[usize])->Tensor{
        let data = vec![0.0;  Tensor::calc_tensor_len_from_shape(shape)];
        Tensor::new(data, shape).unwrap()
    }

    pub fn ones(shape: &[usize])->Tensor{
        let data = vec![1.0;  Tensor::calc_tensor_len_from_shape(shape)];
        Tensor::new(data, shape).unwrap()
    }

    pub fn normal(shape: &[usize], mean: f32, std: f32) -> Result<Tensor, TensorError> {
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = rand::thread_rng();

        let tensor_len = Tensor::calc_tensor_len_from_shape(shape);
        let mut new_data: Vec<f32> = Vec::with_capacity(tensor_len);

        for _ in 0..tensor_len {
            new_data.push(normal.sample(&mut rng));
        }

        Tensor::new(new_data, shape)
    }

    pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Tensor, TensorError> {
        let uniform = Uniform::from(low..high);
        let mut rng = rand::thread_rng();

        let tensor_len = Tensor::calc_tensor_len_from_shape(shape);
        let mut new_data: Vec<f32> = Vec::with_capacity(tensor_len);

        for _ in 0..tensor_len {
            new_data.push(uniform.sample(&mut rng));
        }

        Tensor::new(new_data, shape)
    }

}


impl Mul<f32> for Tensor{
    type Output = Self;
    fn mul(self, other: f32)->Self{
        Tensor::new(self.data.iter().map(|x|x*other).collect(), &self.shape[..]).unwrap()
    }
}

impl MulAssign<f32> for Tensor{
    fn mul_assign(&mut self, rhs: f32) {
        for elem in self.data.iter_mut(){
            *elem *= rhs
        }
    }
}

impl Mul<Tensor> for f32{
    type Output = Tensor;

    fn mul(self, other: Tensor)->Tensor{
        other * self
    }
}

impl<Idx> std::ops::Index<Idx> for Tensor
where
    Idx: std::slice::SliceIndex<[f32]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[index]
    }
}

