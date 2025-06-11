use nalgebra::{Const, DMatrix, OMatrix, SMatrix, SVector};

use crate::initialization::kaiming_weights;

pub struct Linear<const I: usize, const O: usize> {
    w: SMatrix<f32, I, O>,
    b: Option<SVector<f32, O>>,
    cache: Option<(DMatrix<f32>, DMatrix<f32>)>, // x, y after forward, dL/dw dL/db
}

impl<const I: usize, const O: usize> Linear<I, O> {
    pub fn new() -> Self {
        Self {
            w: kaiming_weights(),
            b: Some(SVector::<f32, O>::from_element(0.0f32)),
            cache: None,
        }
    }

    pub fn forward<const B: usize>(
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
    pub fn backwards<const B: usize>(
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
            Err(3usize)
        }
    }

    pub fn optimize(&mut self, lr: f32) -> Result<(), usize> {
        if let Some((dldw, dldb)) = &mut self.cache {
            let dldw = SMatrix::<f32, I, O>::from_row_slice(dldw.as_slice());
            self.w -= dldw * lr;
            if let Some(bias) = &mut self.b {
                let dldb = SMatrix::<f32, O, 1>::from_row_slice(dldb.as_slice());
                *bias -= dldb * lr;
            }
        }
        Ok(())
    }
}

impl<const I: usize, const O: usize> Default for Linear<I, O> {
    fn default() -> Self {
        Self::new()
    }
}
