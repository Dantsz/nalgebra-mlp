use crate::activations::RELU;
use crate::linear::Linear;
use losses::MSELoss;
use nalgebra::{Const, OMatrix, SMatrix};
use rand::Rng;
use std::fs::File;
use std::io::Write;
pub mod activations;
pub mod linear;
pub mod losses;
pub mod sequential;

//Function to approximate
fn aproximate<const B: usize>(input: &SMatrix<f32, B, 2>) -> SMatrix<f32, B, 1> {
    let mut y = SMatrix::<f32, B, 1>::from_element(0.0f32);
    for (i, row) in input.row_iter().enumerate() {
        y[(i, 0)] = 3.0f32 * row[(i, 0)] + 2.0f32 * row[(i, 1)];
    }
    y
}

create_sequential! {SimpleSequential, 2 => l0: Linear<2, 50> => l1: RELU<50> => l2: Linear<50, 1> => 1}

fn main() {
    let mut model = SimpleSequential::default();

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
