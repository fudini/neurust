use std::io::prelude::*;
use std::fs::File;
use byteorder::{ ByteOrder, BigEndian };
use std::io::SeekFrom;
use std::io::BufReader;
use std::ops::Shr;

#[derive(Debug)]
struct MnistHeader {
    magic: u32,
    images_num: u32,
    width: u32,
    height: u32,
}

#[derive(Debug)]
pub struct MnistImage {
    pub digit: u8,
    pub pixels: Vec<f64>,
}

#[derive(Debug)]
pub struct MnistImages {
    pub width: u32,
    pub height: u32,
    pub images: Vec<MnistImage>,
}

fn get_mnist_header(path: &str) -> MnistHeader {

    let mut file = File::open(path).unwrap();
    let mut buffer = [0; 16];

    file.read_exact(&mut buffer);
    let mut slices = buffer.chunks(4);

    MnistHeader {
        magic: BigEndian::read_u32(&slices.next().unwrap()),
        images_num: BigEndian::read_u32(&slices.next().unwrap()),
        width: BigEndian::read_u32(&slices.next().unwrap()),
        height: BigEndian::read_u32(&slices.next().unwrap()),
    }
}

fn bytes_to_floats(buf: &[u8], downsample: bool) -> Vec<f64> {

    let size = 28;

    let mut result: Vec<f64> = vec!();

    if downsample {

        for y in 0..(size / 2) {
            for x in 0..(size / 2) {

                let index = (y * size * 2) + (x * 2);

                // average pixels
                let pixel = (buf[index] as f64
                    + buf[index + 1] as f64
                    + buf[index + size] as f64
                    + buf[index + size + 1] as f64)
                    / (4.0 * 256.0);

                result.push(pixel);
            }
        }
        result
    } else {
        buf.iter().map(|v| (*v as f64) / 256.0).collect::<Vec<f64>>()
    }
}

pub fn load_images(images_path: &str, labels_path: &str, downsample: bool) -> MnistImages {
    
    let mnist_header = get_mnist_header(images_path);
    let image_size = mnist_header.width * mnist_header.height;

    let images_file = File::open(images_path).unwrap();
    let mut images_reader = BufReader::with_capacity(image_size as usize, images_file);

    let labels_file = File::open(labels_path).unwrap();
    let mut labels_reader = BufReader::with_capacity(1, labels_file);

    let mut images: Vec<MnistImage> = vec!();

    for i in 0..mnist_header.images_num {
       
        images_reader.seek(SeekFrom::Start(16 + (i * image_size) as u64));
        let image_bytes = images_reader.fill_buf().unwrap();
        let floats = bytes_to_floats(image_bytes, downsample);

        labels_reader.seek(SeekFrom::Start(8 + i as u64));
        let label_byte = labels_reader.fill_buf().unwrap()[0];

        images.push(MnistImage {
            digit: label_byte,
            pixels: floats,
        });
      
    }
    
    if downsample {

        MnistImages {
            width: mnist_header.width / 2,
            height: mnist_header.height / 2,
            images: images,
        }

    } else {

        MnistImages {
            width: mnist_header.width,
            height: mnist_header.height,
            images: images,
        }
    }
}

pub fn digit_to_outputs(digit: u8, outputs: &mut Vec<f64>){

    for i in 0..outputs.len() {
        if i == digit as usize {
            outputs[i] = 1.0;
        } else {
            outputs[i] = 0.0;
        }
    }
}

pub fn outputs_to_digit(outputs: &Vec<f64>) -> Option<usize> {
    outputs.iter().position(|&r| r > 0.5)
}

pub fn render_digit(pixels: &Vec<f64>) {

    let greys = ['·', '.', ':', '-', '+', '*', '#', '@', '@'];
    let size = (pixels.len() as f64).sqrt() as usize;

    for y in 0..size {

        let mut line = String::from("");
        
        for x in 0..size {
            let i = (y * size) + x;
            let grey_index = ((pixels[i] * 256.0) as usize).shr(5);
            
            line.push(greys[grey_index]);
        }

        println!("{}", line);
    }
}