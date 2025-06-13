use mnist::{Mnist, MnistBuilder};
use nalgebra::{Const, OMatrix, RawStorage, SMatrix};
use nalgebra_mlp::activations::{RELU, Sigmoid};
use nalgebra_mlp::create_sequential;
use nalgebra_mlp::linear::Linear;
use nalgebra_mlp::losses::MSELoss;
use rand::{Rng, rng};

fn to_mat<const W: usize, const H: usize>(data: &[u8]) -> Option<OMatrix<f32, Const<W>, Const<H>>> {
    let data_len = W * H;
    if data.len() != data_len {
        return None;
    }
    let data: Vec<_> = data.iter().map(|x| (*x as f32) / 255.0f32).collect();
    Some(OMatrix::<f32, Const<W>, Const<H>>::from_row_slice(&data))
}

type MnistInstance = (OMatrix<f32, Const<1>, Const<784>>, u8);

create_sequential!(AutoEncoder, 784 => l0:Linear<784,128> => l1:RELU<128> => l2:Linear<128, 64> => l3:RELU<64> => l4:Linear<64,128> => l5:RELU<128> => l6:Linear<128,784> => l7:Sigmoid<784> => 784);

fn train<const B: usize>(model: &mut AutoEncoder, batch: &SMatrix<f32, B, 784>) -> f32 {
    let mut rng = rng();
    let noise_vec: Vec<f32> = (0..B * 784).map(|_| rng.random()).collect();
    let noise = SMatrix::<f32, B, 784>::from_row_slice(&noise_vec);
    let noised = batch + noise;
    let loss = MSELoss::new();
    let recons = model.forward(&noised);
    let l = loss.forward(&recons, batch);
    let l_back = loss.backward(&recons, batch);
    model.backwards(l_back).unwrap();
    model.optimize(0.01f32).unwrap();
    l
}

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
        train_data.push((image, label));
    }

    for i in 0..(TEST_SET as usize) {
        let image = to_mat::<1, 784>(&tst_img[i * 784..(i + 1) * 784]).unwrap();
        let label = tst_lbl[i];
        test_data.push((image, label));
    }

    let mut model = AutoEncoder::default();
    let mut total_train_loss = 0.0f32;
    for (step, (img, _)) in train_data.iter().enumerate() {
        let l = train(&mut model, img);
        total_train_loss += l;
    }
    println!("Train loss: {}", total_train_loss / (TRAIN_SET as f32));

    let mut total_test_loss = 0.0f32;
    for (step, (img, _)) in test_data.iter().enumerate() {
        let loss = MSELoss::default();
        let recon = model.forward(img);
        total_test_loss += loss.forward(&recon, img);
    }
    println!("Test loss: {}", total_test_loss / (TEST_SET as f32));
}
