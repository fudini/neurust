pub mod activations {

    use std::f64::consts::E;

    pub type ActivationFn = (fn(f64) -> f64, fn(f64) -> f64);

    fn sigmoid_activation(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn sigmoid_derivative(o: f64) -> f64 {
        let o = sigmoid_activation(o);
        o * (1.0 - o)
    }

    pub static SIGMOID: ActivationFn = (sigmoid_activation, sigmoid_derivative);
}

pub mod errors {

    pub type ErrorFn = (fn(f64, f64) -> f64, fn(f64, f64) -> f64);

    fn square_error(guess: f64, actual: f64) -> f64 {
        (guess - actual).powf(2.0) / 2.0
    }

    fn square_derivative(guess: f64, actual: f64) -> f64 {
        guess - actual
    }

    pub static SQUARE: ErrorFn = (square_error, square_derivative);
}