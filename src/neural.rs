use rand::Rng;
use rand;
use std::f64::consts::E;
use std::rc::Rc;
use std::cell::RefCell;
use std::mem;

fn activate(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn error(guess: f64, actual: f64) -> f64 {
    (guess - actual).powf(2.0) / 2.0
}

fn derivative(o: f64) -> f64 {
    let o = activate(o);
    o * (1.0 - o)
}

fn error_derivative(guess: f64, actual: f64) -> f64 {
    guess - actual
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    new_weights: Vec<f64>,
    bias: f64,
    total_input: f64,
    input_der: f64,
    output_der: f64,
    output: f64
}

pub trait NeuralValue {
    fn get_value(&self) -> f64;
}

impl NeuralValue for RefCell<Neuron> {
    fn get_value(&self) -> f64 {
        self.borrow().output
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
            .map(|_| rng.gen::<f64>() - 0.5)
            .collect();

        Neuron::with_weights(weights, 0.1)
    }

    // number of weights per neuron has to be the same as number of neurons
    // in the previous layer, or inputs if it's a first layer.
    pub fn with_weights(weights: Vec<f64>, bias: f64) -> Neuron {

        let new_weights = weights.clone();

        Neuron {
            bias: bias,
            total_input: 0f64,
            input_der: 0f64,
            output_der: 0f64,
            output: 0f64,
            weights: weights,
            new_weights: new_weights,
        }
    }

    pub fn feed_forward<T>(&mut self, inputs: &Vec<T>)
        where T: NeuralValue {

        let total_input: f64 = self.weights
            .iter()
            .zip(inputs.iter().map(|input_value| input_value.get_value()))
            .map(|(weight, input)| weight * input)
            .sum();
            
        self.total_input = total_input + self.bias;
        self.output = activate(self.total_input);
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

        if layers.len() < 2 {
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

    pub fn get_outputs(&self) -> Vec<f64> {

        self.layers.last()
            .unwrap()
            .iter()
            .map(|neuron| neuron.get_value())
            .collect::<Vec<f64>>()
    }

    pub fn train(&self, inputs: &Vec<f64>, outputs: &Vec<f64>) -> f64 {

        let network_outputs = self.feed_forward(inputs);

        let last_layer = &network_outputs;

        let mut total_error = 0.0;

        for output_index in 0..last_layer.len() {

            let mut output_neuron = network_outputs[output_index].borrow_mut();
            total_error += error(output_neuron.output, outputs[output_index]);           
            output_neuron.output_der = error_derivative(output_neuron.output, outputs[output_index]);
        }

        for index in (0..self.layers.len()).rev() {

            let current_layer = &self.layers[index];

            for current_index in 0..current_layer.len() {

                let mut current_neuron = current_layer[current_index].borrow_mut();
                current_neuron.input_der = current_neuron.output_der * derivative(current_neuron.total_input);;

                if index >= 1 {

                    for weight_index in 0..current_neuron.weights.len() {

                        let weight = current_neuron.weights[weight_index];
                        let output = self.layers[index - 1][weight_index].get_value();
                        let error_der = current_neuron.input_der * output;
                        current_neuron.new_weights[weight_index] = weight - self.learning_rate * error_der;

                    }
                } else {

                    for weight_index in 0..current_neuron.weights.len() {

                        let weight = current_neuron.weights[weight_index];
                        let output = inputs[weight_index];
                        
                        let error_der = current_neuron.input_der * output;
                        current_neuron.new_weights[weight_index] = weight - self.learning_rate * error_der;
                    }
                }

                current_neuron.bias -= self.learning_rate * current_neuron.input_der;

            }

            if index > 0 {

                let previous_layer = &self.layers[index - 1];

                for previous_neuron_index in 0..previous_layer.len() {
                    
                    let mut previous_neuron = previous_layer[previous_neuron_index].borrow_mut();

                    let mut output_der = 0.0;
                    for current_neuron in current_layer {
                        output_der += current_neuron.borrow().weights[previous_neuron_index] * current_neuron.borrow().input_der;
                    }
                    previous_neuron.output_der = output_der;
                }
            }

        }

        self.swap_weights();

        total_error

    }
}

#[test]
fn test() {

    let inputs = vec!(0.2, 0.3);
    let outputs = vec!(0.0, 1.0, 0.0);

    let mut nn = NeuralNetwork::with_layers(vec!(
        vec!(
            RefCell::new(Neuron::with_weights(vec!(0.2, 0.25), 0.1)),
            RefCell::new(Neuron::with_weights(vec!(0.3, 0.35), 0.1)),
            RefCell::new(Neuron::with_weights(vec!(-0.3, -0.35), 0.1)),
            RefCell::new(Neuron::with_weights(vec!(-0.3, -0.35), 0.1)),
        ),
        vec!(
            RefCell::new(Neuron::with_weights(vec!(0.4, -0.45, 0.5, -0.55), 0.1)),
            RefCell::new(Neuron::with_weights(vec!(0.5, -0.55, 0.6, -0.65), 0.1)),
            RefCell::new(Neuron::with_weights(vec!(0.6, -0.65, 0.4, -0.45), 0.1)),
        ),
    ), 0.5);
    
    nn.train(&inputs, &outputs);
    nn.feed_forward(&vec!(0.1, 0.4));

    let mut network_outputs = nn.get_outputs();
    assert!(network_outputs == vec!(0.4763176074370886, 0.5411613536686533, 0.47558085220979524));

    nn.learning_rate = 0.4;
    nn.train(&inputs, &outputs);
    nn.feed_forward(&vec!(0.1, 0.4));
    network_outputs = nn.get_outputs();
    assert!(network_outputs == vec!(0.45083786140605075, 0.5639480600777952, 0.45002933641579157));

    nn.learning_rate = 0.3;
    nn.train(&inputs, &outputs);
    nn.feed_forward(&vec!(0.1, 0.4));
    network_outputs = nn.get_outputs();
    assert!(network_outputs == vec!(0.43299515834212265, 0.5799061048726579, 0.43214702784312253));

}
