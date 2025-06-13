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

    pub fn optimize(&mut self, _: f32) -> Result<(), usize> {
        Ok(())
    }
}

impl<const I: usize> Default for RELU<I> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Sigmoid<const I: usize> {
    cache: Option<OMatrix<f32, Dyn, Const<I>>>,
}

impl<const I: usize> Sigmoid<I> {
    pub fn new() -> Self {
        Self { cache: None }
    }

    fn sigmoid_fn(x: f32) -> f32 {
        1.0f32 / (1.0f32 + f32::exp(-x))
    }

    pub fn forward<const B: usize>(
        &mut self,
        x: &OMatrix<f32, Const<B>, Const<I>>,
    ) -> OMatrix<f32, Const<B>, Const<I>> {
        self.cache = Some(x.resize_vertically(B, 0.0f32)); // no actual resize

        x.clone_owned().map(Self::sigmoid_fn)
    }

    pub fn backwards<const B: usize>(
        &mut self,
        dldy: OMatrix<f32, Const<B>, Const<I>>,
    ) -> Result<OMatrix<f32, Const<B>, Const<I>>, usize> {
        if let Some(x) = &mut self.cache {
            let dydx = x.map(|x| Self::sigmoid_fn(x) * (1.0f32 - Self::sigmoid_fn(x)));
            let dldx = dldy.component_mul(&dydx);
            Ok(dldx)
        } else {
            Err(1)
        }
    }

    pub fn optimize(&mut self, _: f32) -> Result<(), usize> {
        Ok(())
    }
}

impl<const I: usize> Default for Sigmoid<I> {
    fn default() -> Self {
        Self::new()
    }
}
