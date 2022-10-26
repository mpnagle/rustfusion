extern crate ndarray;
extern crate mnist;
extern crate image;

use image::RgbImage;
use mnist::*;
use ndarray::prelude::*;


fn array_to_image(arr: Array3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (height, width, _) = arr.dim();
    let raw = arr.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

fn main() {

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
     tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);


        
    let mut array: Array3<u8> = Array3::zeros((200, 250, 3)); // 250x200 RGB
    
    for ((x, y, z), v) in array.indexed_iter_mut() {
        *v = match z {
            0 => y as u8,
            1 => x as u8,
            2 => 0,
            _ => unreachable!(),
        };
    }

    let image = array_to_image(array);
    image.save("out.png");
}