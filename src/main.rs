extern crate rand;
extern crate time;

mod neural;
use neural::NeuralNetwork;
use time::now;


pub fn main() {

    let input1 = (vec!(1.0, 1.0), vec!(0.0));
    let input2 = (vec!(0.0, 1.0), vec!(1.0));
    let input3 = (vec!(1.0, 0.0), vec!(1.0));
    let input4 = (vec!(0.0, 0.0), vec!(0.0));

    let mut nn = NeuralNetwork::new(vec!(2, 4, 1), 1.0);

    let mut error = 0.0;

    let start_time = now();

    for _ in 0..100 {

        for _ in 0..1000 {
            nn.train(&input1.0, &input1.1);
            nn.train(&input2.0, &input2.1);
            nn.train(&input3.0, &input3.1);
            error = nn.train(&input4.0, &input4.1);
        }

        nn.learning_rate *= 0.99;
        println!("Error: {:?} Rate: {:?}", error, nn.learning_rate);

    }

    println!("Total time: {:?}", now() - start_time);

    nn.feed_forward(&input1.0);
    println!("1 1 => {:#?}", nn.get_activations());

    nn.feed_forward(&input2.0);
    println!("0 1 => {:#?}", nn.get_activations());
    
    nn.feed_forward(&input3.0);
    println!("1 0 => {:#?}", nn.get_activations());
    
    nn.feed_forward(&input4.0);
    println!("0 0 => {:#?}", nn.get_activations());
}