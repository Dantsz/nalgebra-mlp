use mnist::{Mnist, MnistBuilder};
use nalgebra::{Const, OMatrix, RawStorage};
use nalgebra_mlp::create_sequential;
use std::fs::OpenOptions;
use std::io::Write;

fn to_mat<const W: usize, const H: usize>(data: &[u8]) -> Option<OMatrix<f32, Const<W>, Const<H>>> {
    let data_len = W * H;
    if data.len() != data_len {
        return None;
    }
    let data: Vec<_> = data.iter().map(|x| *x as f32).collect();
    Some(OMatrix::<f32, Const<W>, Const<H>>::from_row_slice(&data))
}

type MnistInstance = (OMatrix<f32, Const<1>, Const<784>>, u8);

//TODO: implement sigmoid
//TODO: defined autoencoder
//TODO: run
fn main() {
    const TRAIN_SET: u32 = 60_000;
    const TEST_SET: u32 = 10_000;
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit() // Labels will be digits 0-9
        .training_set_length(TRAIN_SET)
        .test_set_length(TEST_SET)
        .base_url("https://raw.githubusercontent.com/fgnt/mnist/master")
        .download_and_extract()
        .finalize();

    let mut train_data: Vec<MnistInstance> = Vec::new();
    let mut test_data: Vec<MnistInstance> = Vec::new();

    for i in 0..(TRAIN_SET as usize) {
        let image = to_mat::<1, 784>(&trn_img[i * 784..(i + 1) * 784]).unwrap();
        let label = trn_lbl[i];
    }

    for i in 0..(TEST_SET as usize) {
        let image = to_mat::<1, 784>(&tst_img[i * 784..(i + 1) * 784]).unwrap();
        let label = tst_lbl[i];
    }
    unimplemented!()
}
