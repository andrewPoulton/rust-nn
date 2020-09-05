
#![allow(unused)]
#![macro_use]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_nn::{mmul_, linlayer ,Tensor};

pub fn mmul_benchmark(c: &mut Criterion) {
    c.bench_function("mat mul", |b| b.iter(|| mmul_()));
}

pub fn lin_layer_benchmark(c: &mut Criterion) {
    c.bench_function("linear layer", |b| b.iter(|| linlayer()));
}

criterion_group!(benches, mmul_benchmark, lin_layer_benchmark);
criterion_main!(benches);

