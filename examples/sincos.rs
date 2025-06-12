use nalgebra::SMatrix;
use nalgebra_mlp::activations::RELU;
use nalgebra_mlp::create_sequential;
use nalgebra_mlp::linear::Linear;
use nalgebra_mlp::losses::MSELoss;
use rand::Rng;
//Function to approximate
fn to_be_approximated<const B: usize>(input: &SMatrix<f32, B, 2>) -> SMatrix<f32, B, 1> {
    let mut y = SMatrix::<f32, B, 1>::from_element(0.0f32);
    for (i, row) in input.row_iter().enumerate() {
        y[(i, 0)] = 3.0f32 * f32::cos(row[(i, 0)]) + 2.0f32 * f32::sin(row[(i, 1)]);
    }
    y
}

// Define a very simple feed-forward
create_sequential! {SimpleSequential, 2 => l0: Linear<2, 64> => l1: RELU<64> => l2: Linear<64, 1>  => 1}

fn main() -> Result<(), usize> {
    let mut rng = rand::rng();
    let mut model = SimpleSequential::default();
    const STEPS: usize = 10_000usize;
    let loss = MSELoss::new();

    //Train model
    for i in 0..STEPS {
        let (x, y) = rng.random::<(f32, f32)>();
        let input = SMatrix::<f32, 1, 2>::from_row_slice(&[x, y]);
        let y = model.forward(&input);
        let target = to_be_approximated(&input);
        let l = loss.forward(&y, &target);
        println!("step:{i} y: {:?}, target: {:?}, loss: {:?}", y, target, l);
        let l_back = loss.backward(&y, &target);
        model.backwards(l_back)?;
        model.optimize(0.0005f32)?;
    }
    //Test model
    let input = SMatrix::<f32, 1, 2>::from_row_slice(&[0.10f32, 0.25f32]);
    let y = model.forward(&input);
    let control = to_be_approximated(&input);
    println!(
        "y: {:?} control: {:?} MSE:{:?}",
        y,
        control,
        loss.forward(&y, &control)
    );
    Ok(())
}
