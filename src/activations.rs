use nalgebra::{Const, Dyn, OMatrix};

pub struct RELU<const I: usize> {
    cache: Option<OMatrix<f32, Dyn, Const<I>>>,
}

impl<const I: usize> RELU<I> {
    pub fn new() -> Self {
        RELU { cache: None }
    }

    pub fn forward<const B: usize>(
        &mut self,
        x: &OMatrix<f32, Const<B>, Const<I>>,
    ) -> OMatrix<f32, Const<B>, Const<I>> {
        self.cache = Some(x.resize_vertically(B, 0.0f32)); // no actual resize

        x.clone_owned().map(|e| f32::max(0.0f32, e))
    }

    pub fn backwards<const B: usize>(
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
