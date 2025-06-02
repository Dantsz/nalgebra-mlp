use nalgebra::{Const, DMatrix, Dyn, OMatrix, SMatrix, SVector};

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
        x: SMatrix<f32, B, I>,
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
        let y = x.clone_owned().map(|e| f32::max(0.0f32, e));
        self.cache = Some(y.resize_vertically(B, 0.0f32)); // no actual resize
        y
    }

    fn backwards(
        &mut self,
        dldy: OMatrix<f32, Dyn, Const<I>>,
    ) -> Result<OMatrix<f32, Dyn, Const<I>>, usize> {
        if let Some(x) = &mut self.cache {
            let dydx = x.map(|e| if e > 0.0f32 { 1.0f32 } else { 0.0f32 });
            let dldx = dldy.component_mul(&dydx);
            Ok(dldx)
        } else {
            Err(1)
        }
    }
}

fn main() {
    let mut layer0 = Linear::<2, 2>::new();
    let mut relu0 = RELU::<2>::new();

    let x = SMatrix::<f32, 1, 2>::repeat(1.0f32);

    let y = layer0.forward(x);
    let ny = relu0.forward(&y);

    let dldy = SMatrix::<f32, 1, 2>::from_element(1.0f32);
    let dldx = layer0.backwards(dldy);
    println!("Result of forward pass: {:?}", y);
    println!("Result of backwards pass: {:?}", dldx);
    println!("Layer cache: {:?}", layer0.cache);
}
