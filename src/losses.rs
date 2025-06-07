use nalgebra::{Const, OMatrix};

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }

    // Computes the scalar MSE loss: mean((pred - target)^2)
    pub fn forward<const B: usize, const I: usize>(
        &self,
        pred: &OMatrix<f32, Const<B>, Const<I>>,
        target: &OMatrix<f32, Const<B>, Const<I>>,
    ) -> f32 {
        let diff = pred - target;
        let squared_error = diff.map(|x| x * x);
        let sum: f32 = squared_error.iter().sum();
        sum / (B * I) as f32
    }

    // Computes the gradient of the MSE loss w.r.t. pred: 2 * (pred - target) / (B * I)
    pub fn backward<const B: usize, const I: usize>(
        &self,
        pred: &OMatrix<f32, Const<B>, Const<I>>,
        target: &OMatrix<f32, Const<B>, Const<I>>,
    ) -> OMatrix<f32, Const<B>, Const<I>> {
        let scale = 2.0 / (B * I) as f32;
        (pred - target) * scale
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}
