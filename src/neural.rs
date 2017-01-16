use rand::Rng;
use rand;
use std::f64::consts::E;
use std::rc::Rc;
use std::cell::RefCell;
use std::mem;

fn activate(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn error(guess: &f64, actual: &f64) -> f64 {
    (guess - actual).powf(2.0) / 2.0
}

fn derivative(o: f64) -> f64 {
    o * (1.0 - o)
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    new_weights: Vec<f64>,
    bias: f64,
    error: f64,
    activation: f64,
    derivative: f64
}

pub trait NeuralValue {
    fn get_value(&self) -> f64;
}

impl NeuralValue for RefCell<Neuron> {
    fn get_value(&self) -> f64 {
        self.borrow().activation
    }
}

impl NeuralValue for f64 {
    fn get_value(&self) -> f64 {
        *self
    }
}

impl Neuron {

    pub fn new(inputs_num: u16) -> Neuron {

        let mut rng = rand::thread_rng();

        let weights: Vec<f64> = (0..inputs_num)
            .into_iter()
            .map(|_| rng.gen::<f64>())
            .collect();

        Neuron::with_weights(weights, rng.gen::<f64>() * 2.0 - 1.0)
    }

    // number of weights per neuron has to be the same as number of neurons
    // in the previous layer, or inputs if it's a first layer.
    pub fn with_weights(weights: Vec<f64>, bias: f64) -> Neuron {

        let new_weights = weights.clone();

        Neuron {
            bias: bias,
            error: 0f64,
            activation: 0f64,
            derivative: 0f64,
            weights: weights,
            new_weights: new_weights,
        }
    }

    pub fn feed_forward<T>(&mut self, inputs: &Vec<T>)
        where T: NeuralValue {

        let sum: f64 = self.weights
            .iter()
            .zip(inputs.iter().map(|input_value| input_value.get_value()))
            .map(|(weight, input)| weight * input)
            .sum();

        self.activation = activate(sum + self.bias);
    }

    pub fn swap_weights(&mut self) {
        mem::swap(&mut self.weights, &mut self.new_weights);
    }

}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Rc<Vec<Vec<RefCell<Neuron>>>>,
    pub learning_rate: f64,
}

impl NeuralNetwork {

    pub fn new(config: Vec<u16>, learning_rate: f64) -> NeuralNetwork {

        if config.len() < 3 {
            panic!("Your network should have at least 1 input, hidden and output layers.");
        }

        let mut input_count = config[0];
        let mut layers = vec!();

        for layer_config in config.iter().skip(1) {
            let layer = (0..*layer_config).map(|_| RefCell::new(Neuron::new(input_count))).collect();
            layers.push(layer);
            input_count = *layer_config;
        }

        NeuralNetwork {
            layers: Rc::new(layers),
            learning_rate: learning_rate
        }
    }

    pub fn with_layers(layers: Vec<Vec<RefCell<Neuron>>>, learning_rate: f64) -> NeuralNetwork {

        if layers.len() < 3 {
            panic!("Your network should have at least 1 input, hidden and output layers.");
        }

        NeuralNetwork {
            layers: Rc::new(layers),
            learning_rate: learning_rate
        }
    }

    pub fn feed_forward<T>(&self, inputs: &Vec<T>) -> &Vec<RefCell<Neuron>>
        where T: NeuralValue {

        let layers = &self.layers;

        // feed first layer from inputs
        for neuron in &layers[0] {
            neuron.borrow_mut().feed_forward(&inputs);
        }

        // feed each layer from activations in the previous layer
        for index in 1..layers.len() {

            let previous_layer = &layers[index - 1];
            let current_layer = &layers[index];

            for neuron in current_layer {
                neuron.borrow_mut().feed_forward(previous_layer);
            }
        }

        &layers[layers.len() - 1]
    }

    pub fn swap_weights(&self) {

        for layer_index in 0..self.layers.len() {
            for neuron in &self.layers[layer_index] {
                neuron.borrow_mut().swap_weights();
            }
        }
    }

    pub fn get_activations(&self) -> Vec<f64> {

        self.layers.last()
            .unwrap()
            .iter()
            .map(|neuron| neuron.get_value())
            .collect::<Vec<f64>>()
    }

    pub fn train(&self, inputs: &Vec<f64>, outputs: &Vec<f64>) -> f64 {

        let network_outputs = self.feed_forward(inputs);

        let errors: Vec<f64> = network_outputs
            .iter()
            .zip(outputs)
            .map(|(guess, actual)| error(&guess.get_value(), actual))
            .collect();

        let total_error: f64 = errors.iter().sum();

        let last_layer = &network_outputs;
        let previous_layer = &self.layers[self.layers.len() - 2];

        for output_index in 0..last_layer.len() {

            let output_neuron = &network_outputs[output_index];
            let activation = output_neuron.get_value();
            let output_derivative = derivative(activation);
            let output_error = activation - outputs[output_index];

            for weight_index in 0..previous_layer.len() {

                let weight_activation = previous_layer[weight_index].get_value();
                let weight_delta = output_error * output_derivative * weight_activation;

                let new_weight = output_neuron.borrow().weights[weight_index]
                    - self.learning_rate * weight_delta;

                let mut output_neuron = output_neuron.borrow_mut();

                output_neuron.new_weights[weight_index] = new_weight;
                output_neuron.error = output_error;
                output_neuron.derivative = output_derivative;

            }
        }

        for index in (1..self.layers.len()).rev() {

            let current_layer = &self.layers[index - 1];
            let output_layer = &self.layers[index];

            for current_index in 0..current_layer.len() {

                let out_error = 0.0;

                let out_error = output_layer
                    .iter()
                    .fold(0.0, |err, output_neuron| {
                        let output_neuron = &output_neuron.borrow();
                        let output_error = output_neuron.derivative * output_neuron.error;
                        let output_d = output_error * output_neuron.weights[current_index];
                        err + output_d
                    });

                let mut current_neuron = current_layer[current_index].borrow_mut();
                current_neuron.error = out_error;
                current_neuron.derivative = derivative(current_neuron.activation);

                for weight_index in 0..inputs.len() {

                    let something = out_error * current_neuron.derivative * inputs[weight_index];
                    current_neuron.new_weights[weight_index] = current_neuron.weights[weight_index] - (self.learning_rate * something);
                }
            }
        }

        self.swap_weights();

        total_error
    }
}
