use approx::assert_relative_eq;
use nalgebra::{Const, OMatrix};

fn softmax<const B: usize, const I: usize>(
    input: &OMatrix<f32, Const<B>, Const<I>>,
) -> OMatrix<f32, Const<B>, Const<I>> {
    //TODO: Vec<f32> should be avoided
    let sums: Vec<f32> = input
        .row_iter()
        .map(|row| row.iter().map(|x| f32::exp(*x)).sum())
        .collect();

    let mut elems = OMatrix::<f32, Const<B>, Const<I>>::from_element(0.0f32);
    for i in 0..B {
        for j in 0..I {
            elems[(i, j)] = f32::exp(input[(i, j)]) / sums[i];
        }
    }

    elems
}

#[test]
fn test_softmax_basic() {
    let test_input = OMatrix::<f32, Const<2>, Const<2>>::new(24.0f32, 5.0f32, -12.0f32, 25.0f32);
    let output = softmax(&test_input);

    assert!(
        output[(0, 0)] > output[(0, 1)],
        "assert: {} > {} failed",
        output[(0, 0)],
        output[(0, 1)]
    );
    for i in 0..1usize {
        assert_relative_eq!(output[(i, 0)] + output[(i, 1)], 1.0f32);
    }
}
