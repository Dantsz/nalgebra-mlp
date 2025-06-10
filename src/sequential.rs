//Sequential macro to chain layers/activation/etc in a type

#[macro_export]
macro_rules! create_sequential {
    ($name:ident, $input_size: literal => $($module_label:ident :$module_type:ty)=>+ => $output_size: literal) => {
        paste::paste! {
            #[derive(Default)]
            pub struct $name {
                $(
                    pub $module_label: $module_type
                ),*
            }
            impl $name {
                pub fn forward<const B: usize>(&mut self, x: &nalgebra::SMatrix<f32, B, $input_size>) -> nalgebra::SMatrix<f32, B, $output_size> {
                    struct Pipeline<T>(T);

                    impl<T> Pipeline<T> {
                        fn pipe<F, R>(self, f: F) -> Pipeline<R>
                        where
                            F: FnOnce(&T) -> R,
                        {
                            Pipeline(f(&self.0))
                        }

                        fn unwrap(self) -> T {
                            self.0
                        }
                    }
                    Pipeline(*x)$(.pipe(|x| self.$module_label.forward(x)))*.unwrap()
                }
                create_sequential!{@GenerateBackwards, $input_size, $output_size, $($module_label),*}

                pub fn optimize(&mut self, lr: f32) -> Result<(), usize> {
                    $(
                        self.$module_label.optimize(lr)?;
                    )*
                    Ok(())
                }
            }
        }
    };
    (@GenerateBackwards, $input_size: literal, $output_size: literal, $module:ident, $grad_provider:ident, $($tail:ident),*) => {
        create_sequential!{@GenerateBackwards, $input_size, $output_size, $module, [$module;$grad_provider], $grad_provider, $($tail),*}
    };
    (@GenerateBackwards, $input_size: literal, $output_size: literal, $first_module: ident,
    [$($processed:ident;$processed_grad:ident),*], $module:ident, $grad_provider:ident, $($tail:ident),*) => {
        create_sequential!{@GenerateBackwards, $input_size, $output_size, $first_module, [$module;$grad_provider, $($processed;$processed_grad),*], $grad_provider, $($tail),*}
    };
    (@GenerateBackwards, $input_size: literal, $output_size: literal, $first_module: ident,
    [$($processed:ident;$processed_grad:ident),*], $module:ident, $grad_provider:ident) => {
        create_sequential!{@GenerateBackwards, $input_size, $output_size, $first_module, [$module;$grad_provider, $($processed;$processed_grad),*], $grad_provider}
    };

    (@GenerateBackwards, $input_size: literal, $output_size: literal, $first_module: ident,
    [$($processed:ident;$grad_provider:ident),*], $last_label:ident) => {
        paste::paste!{
            fn backwards<const B: usize>(&mut self, dldy: nalgebra::SMatrix<f32, B, $output_size>) -> Result<nalgebra::SMatrix<f32, B, $input_size>, usize> {
                let [<dldx_$last_label>] = self.$last_label.backwards(dldy)?;
                $(
                    let [<dldx_$processed>] = self.$processed.backwards([<dldx_$grad_provider>])?;
                )*
                Ok([<dldx_$first_module>])
            }
        }
    };
}
