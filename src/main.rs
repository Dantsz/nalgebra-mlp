use nalgebra::{Const, DMatrix, Dyn, OMatrix, SMatrix, SVector};
use rand::Rng;
use std::fs::File;
use std::io::Write;

struct Linear<const I: usize, const O: usize> {
    w: SMatrix<f32, I, O>,
    b: Option<SVector<f32, O>>,
    cache: Option<(DMatrix<f32>, DMatrix<f32>)>, // x, y after forward, dL/dw dL/db
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn new() -> Self {
        Self {
            w: SMatrix::<f32, I, O>::identity(),
            b: Some(SVector::<f32, O>::from_element(0.0f32)),
            cache: None,
        }
    }

    fn forward<const B: usize>(
        &mut self,
        x: &SMatrix<f32, B, I>,
    ) -> OMatrix<f32, Const<B>, Const<O>> {
        let mut mul = x * self.w;
        if let Some(bias) = self.b {
            for mut row in mul.row_iter_mut() {
                row += bias.transpose()
            }
        }
        self.cache = Some((
            DMatrix::from_row_slice(B, I, x.as_slice()),
            DMatrix::from_row_slice(B, O, mul.as_slice()),
        ));
        mul
    }
    // B must match the one used in forward
    // return dldx, adn writes dldw and dldb to cache
    fn backwards<const B: usize>(
        &mut self,
        dldy: SMatrix<f32, B, O>,
    ) -> Result<SMatrix<f32, B, I>, usize> {
        if let Some((x, y)) = &mut self.cache {
            let previous_batch_size = x.nrows();
            if previous_batch_size != B {
                return Err(1usize);
            }

            let dldx = dldy * self.w.transpose();

            let dldw = x.transpose() * dldy;
            if dldw.nrows() != I {
                return Err(2usize);
            }

            let dldb = dldy.row_sum();
            // dLdW
            *x = DMatrix::from_row_slice(I, O, dldw.as_slice());
            // dldb
            *y = DMatrix::from_row_slice(1, O, dldb.as_slice());
            Ok(dldx)
        } else {
            Err(2usize)
        }
    }

    fn optimize(&mut self, lr: f32) -> Result<(), usize> {
        if let Some((dldw, _)) = &mut self.cache {
            let dldw = SMatrix::<f32, I, O>::from_row_slice(dldw.as_slice());
            self.w -= dldw * lr;
        }
        Ok(())
    }
}

struct RELU<const I: usize> {
    cache: Option<OMatrix<f32, Dyn, Const<I>>>,
}

impl<const I: usize> RELU<I> {
    fn new() -> Self {
        RELU { cache: None }
    }

    fn forward<const B: usize>(
        &mut self,
        x: &OMatrix<f32, Const<B>, Const<I>>,
    ) -> OMatrix<f32, Const<B>, Const<I>> {
        self.cache = Some(x.resize_vertically(B, 0.0f32)); // no actual resize

        x.clone_owned().map(|e| f32::max(0.0f32, e))
    }

    fn backwards<const B: usize>(
        &mut self,
        dldy: OMatrix<f32, Const<B>, Const<I>>,
    ) -> Result<OMatrix<f32, Const<B>, Const<I>>, usize> {
        if let Some(x) = &mut self.cache {
            let dydx = x.map(|e| if e > 0.0f32 { 1.0f32 } else { 0.0f32 });
            let dldx = dldy.component_mul(&dydx);
            Ok(dldx)
        } else {
            Err(1)
        }
    }
}

struct SimpleMLP {
    linear0: Linear<2, 2>,
    act0: RELU<2>,
    linear1: Linear<2, 1>,
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

/// Testing

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
    let steps = 10000usize;
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
        model.optimize(0.001f32).expect("Should not fail");
        losses.push(l);
        println!("y: {:?}, target: {:?}, loss: {:?}", y, target, l);
    }

    let mut file = File::create("losses.txt").expect("Should not fail");
    for num in losses {
        writeln!(file, "{}", num).expect("Should not fail");
    }

    let input = SMatrix::<f32, 1, 2>::from_row_slice(&[70.0f32, 25.0f32]);
    let y = model.forward(&input);
    let control = aproximate(&input);
    println!("y: {:?} control: {:?}", y, control);
}
