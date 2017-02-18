extern crate rand;
extern crate time;
extern crate byteorder;

mod neural;
mod mnist;
mod functions;

use neural::{ NeuralNetwork, Neuron };
use time::now;
use mnist::*;
use functions::activations::SIGMOID;
use functions::errors::SQUARE;

fn main() {

    run_mnist();
    // run_xor();
}

fn run_mnist() {

    let downsample = true;

    println!("Loading MNIST learning set...");
    let training_set = load_images("./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte", downsample);
    
    println!("Loading MNIST testing set...");
    let testing_set = load_images("./mnist/t10k-images.idx3-ubyte", "./mnist/t10k-labels.idx1-ubyte", downsample);

    let inputs_num = training_set.width * training_set.height;

    let mut nn = NeuralNetwork::new(vec!(inputs_num as u16, 200, 60, 10), 0.5, SIGMOID, SQUARE);

    let mut outputs: Vec<f64> = (0..10).map(|_| 0.0).collect();
    let mut error = 0.0;

    let iterations = 100;
    let images_count = 6;

    let images_set = 10000;
    
    for iteration in 0..iterations {

        println!("Iteration: {:?}", iteration);

        let start_time = now();

        for i in 0..images_count {

            error = 0.0;

            for j in 0..images_set {
                let image = &training_set.images[j + (i * images_set)];
                digit_to_outputs(image.digit, &mut outputs);
                error += nn.train(&image.pixels, &outputs);
            }

            error = error / images_set as f64;

            println!("Batch: {:?} Error: {:?} Rate: {:?}", i, error, nn.learning_rate);

        }

        println!("Time: {:?} ms", (now() - start_time).num_milliseconds());

        nn.learning_rate *= 0.98;

        let num_predicted = testing_set.images.iter()
            .fold(0, |num, image| {
                
                nn.feed_forward(&image.pixels);

                if outputs_to_digit(&nn.get_outputs()) == Some(image.digit as usize) {
                    num + 1
                } else {
                    num
                }
            });

        println!("Predicted: {:?}", num_predicted);
        println!("");

    }

}

fn run_xor () {

    let input1 = (vec!(1.0, 1.0), vec!(0.0));
    let input2 = (vec!(0.0, 1.0), vec!(1.0));
    let input3 = (vec!(1.0, 0.0), vec!(1.0));
    let input4 = (vec!(0.0, 0.0), vec!(0.0));

    let mut nn = NeuralNetwork::new(vec!(2, 5, 1), 0.5, SIGMOID, SQUARE);

    let mut error = 0.0;

    let start_time = now();

    for i in 0..100 {

        for _ in 0..1000 {
            nn.train(&input1.0, &input1.1);
            nn.train(&input2.0, &input2.1);
            nn.train(&input3.0, &input3.1);
            error = nn.train(&input4.0, &input4.1);
        }

        nn.learning_rate *= 0.99;
        
        println!("Iter: {:?} Error: {:?} Rate: {:?}", i,  error, nn.learning_rate);

    }

    println!("Total time: {:?}", now() - start_time);

    nn.feed_forward(&input1.0);
    println!("1 1 => {:#?}", nn.get_outputs());

    nn.feed_forward(&input2.0);
    println!("0 1 => {:#?}", nn.get_outputs());
    
    nn.feed_forward(&input3.0);
    println!("1 0 => {:#?}", nn.get_outputs());
    
    nn.feed_forward(&input4.0);
    println!("0 0 => {:#?}", nn.get_outputs());
}

