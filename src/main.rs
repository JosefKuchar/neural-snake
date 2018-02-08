extern crate rand;

use rand::Rng;
use rand::distributions::{IndependentSample, Range};

fn main() {
    let mut network = Network::new(&[2, 2, 1]);

    loop {
        let dna = Dna::new(8);

        network.update_weights(dna.genomes);

        let mut results: Vec<f32> = Vec::new();
        results.push(network.run(&[-1f32, -1f32])[0]);
        results.push(network.run(&[-1f32, 1f32])[0]);
        results.push(network.run(&[1f32, -1f32])[0]);
        results.push(network.run(&[1f32, 1f32])[0]);

        if results[0] < 0f32 && results[1] > 0f32 && results[2] > 0f32 && results[3] < 0f32 {
            println!("{:?}", network.weights);
            println!("{:?}", results);
            break;
        }
    }
}

struct Dna {
    genomes: Vec<f32>,
    fitness: f32,
}

impl Dna {
    pub fn new(size: usize) -> Dna {
        let mut genomes: Vec<f32> = Vec::with_capacity(size);
        let mut rng = rand::thread_rng();
        let range = Range::new(-1f32, 1f32);

        for _ in 0..size {
            genomes.push(range.ind_sample(&mut rng));
        }

        return Dna {
            genomes: genomes,
            fitness: 0f32,
        };
    }

    pub fn crossover(&self, partner: &Dna) -> Dna {
        let mut rng = rand::thread_rng();
        let mut genomes: Vec<f32> = Vec::with_capacity(self.genomes.len());

        for (index, genome) in self.genomes.iter().enumerate() {
            if rng.gen() {
                genomes.push(*genome);
            } else {
                genomes.push(partner.genomes[index]);
            }
        }

        return Dna {
            genomes: genomes,
            fitness: 0f32,
        };
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let range = Range::new(0, 100);
        let genome_range = Range::new(-1f32, 1f32);

        for genome in self.genomes.iter_mut() {
            if range.ind_sample(&mut rng) == 0 {
                *genome = genome_range.ind_sample(&mut rng);
            }
        }
    }
}

struct Network {
    weights: Vec<Vec<Vec<f32>>>,
}

impl Network {
    pub fn new(sizes: &[usize]) -> Network {
        let mut layers: Vec<Vec<Vec<f32>>> = Vec::new();
        for i in 0..(sizes.len() - 1) {
            let mut layer: Vec<Vec<f32>> = Vec::new();
            for _ in 0..sizes[i + 1] {
                let mut node: Vec<f32> = Vec::new();
                for _ in 0..sizes[i] {
                    node.push(0f32);
                }

                if i != sizes.len() - 2 {
                    node.push(0f32);
                }

                node.shrink_to_fit();
                layer.push(node);
            }
            layer.shrink_to_fit();
            layers.push(layer);
        }

        return Network { weights: layers };
    }

    pub fn run(&self, input: &[f32]) -> Vec<f32> {
        let mut result: Vec<f32> = input.to_vec();

        for layer in &self.weights {
            let mut current: Vec<f32> = Vec::new();

            for node in layer {
                let mut sum: f32 = 0.0;
                for (index, weight) in node.iter().enumerate() {
                    if index >= result.len() {
                        sum += weight;
                        continue;
                    }

                    sum += result[index] * weight;
                }
                //println!("{}", sum);
                current.push(tanh(sum));
            }

            result = current;
        }

        return result;
    }

    pub fn update_weights(&mut self, weights: Vec<f32>) {
        let mut index = 0;
        for layer in self.weights.iter_mut() {
            for node in layer {
                for weight in node {
                    *weight = weights[index];
                    index += 1;
                }
            }
        }
    }
}

fn sigmoid(y: f32) -> f32 {
    1f32 / (1f32 + (-y).exp())
}

fn tanh(y: f32) -> f32 {
    let a = (2f32 * y).exp();
    return (a - 1f32) / (a + 1f32);
}
