use nalgebra::{Const, OMatrix};
use rand::{
    distr::{Distribution, Uniform},
    rng,
};

pub fn kaiming_weights<const D: usize, const I: usize>() -> OMatrix<f32, Const<D>, Const<I>> {
    let limit = (6.0_f32 / I as f32).sqrt(); // sqrt(6 / fan_in)
    let uniform = Uniform::new(-limit, limit).expect("");
    let mut rng = rng();
    let mut matrix = OMatrix::<f32, Const<D>, Const<I>>::from_element(0.0f32);
    matrix
        .iter_mut()
        .for_each(|x| *x = uniform.sample(&mut rng));
    matrix
}
