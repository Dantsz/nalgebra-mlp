use crate::activations::RELU;
use crate::linear::Linear;
use nalgebra::{Const, OMatrix, SMatrix};
use rand::Rng;
use std::fs::File;
use std::io::Write;
pub mod activations;
pub mod linear;

struct SimpleMLP {
    linear0: Linear<2, 50>,
    act0: RELU<50>,
    linear1: Linear<50, 1>,
}

impl SimpleMLP {
    fn new() -> Self {
        Self {
            linear0: Linear::new(),
            act0: RELU::new(),
            linear1: Linear::new(),
        }
    }
    fn forward<const B: usize>(
        &mut self,
        x: &OMatrix<f32, Const<B>, Const<2>>,
    ) -> OMatrix<f32, Const<B>, Const<1>> {
        let y0 = self.linear0.forward(x);
        let y1 = self.act0.forward(&y0);

        self.linear1.forward(&y1)
    }
    //return dldx
    fn backwards<const B: usize>(
        &mut self,
        dldy: OMatrix<f32, Const<B>, Const<1>>,
    ) -> Result<OMatrix<f32, Const<B>, Const<2>>, usize> {
        let dldx1 = self.linear1.backwards(dldy);
        if dldx1.is_err() {
            return Err(1);
        }
        let dldx0act = self.act0.backwards(dldx1.unwrap());
        if dldx0act.is_err() {
            return Err(2);
        }
        let dldx = self.linear0.backwards(dldx0act.unwrap());
        if dldx.is_err() {
            return Err(3);
        }
        Ok(dldx.unwrap())
    }
    fn optimize(&mut self, lr: f32) -> Result<(), usize> {
        self.linear0.optimize(lr)?;
        self.linear1.optimize(lr)?;
        Ok(())
    }
}

struct MSELoss;

impl MSELoss {
    fn new() -> Self {
        MSELoss
    }

    // Computes the scalar MSE loss: mean((pred - target)^2)
    fn forward<const B: usize, const I: usize>(
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
    fn backward<const B: usize, const I: usize>(
        &self,
        pred: &OMatrix<f32, Const<B>, Const<I>>,
        target: &OMatrix<f32, Const<B>, Const<I>>,
    ) -> OMatrix<f32, Const<B>, Const<I>> {
        let scale = 2.0 / (B * I) as f32;
        (pred - target) * scale
    }
}

// Testing

//Function to approximate
fn aproximate<const B: usize>(input: &SMatrix<f32, B, 2>) -> SMatrix<f32, B, 1> {
    let mut y = SMatrix::<f32, B, 1>::from_element(0.0f32);
    for (i, row) in input.row_iter().enumerate() {
        y[(i, 0)] = 3.0f32 * row[(i, 0)] + 2.0f32 * row[(i, 1)];
    }
    y
}

fn main() {
    let mut model = SimpleMLP::new();
    println!("TEST");
    let mut rng = rand::rng();
    let steps = 2_000usize;
    let loss = MSELoss::new();
    let mut losses = Vec::new();

    for i in 0..steps {
        let x = rng.random::<f32>();
        let y = rng.random::<f32>();
        let input = SMatrix::<f32, 1, 2>::from_row_slice(&[x, y]);
        let y = model.forward(&input);
        let target = aproximate(&input);
        let l = loss.forward(&y, &target);
        let l_back = loss.backward(&y, &target);
        model.backwards(l_back).expect("Should not fail");
        model.optimize(0.01f32).expect("Should not fail");
        losses.push(l);
        println!("step:{i} y: {:?}, target: {:?}, loss: {:?}", y, target, l);
    }

    let mut file = File::create("losses.txt").expect("Should not fail");
    for num in losses {
        writeln!(file, "{}", num).expect("Should not fail");
    }

    let input = SMatrix::<f32, 1, 2>::from_row_slice(&[0.10f32, 0.25f32]);
    let y = model.forward(&input);
    let control = aproximate(&input);
    println!(
        "y: {:?} control: {:?} MSE:{:?}",
        y,
        control,
        loss.forward(&y, &control)
    );
}
