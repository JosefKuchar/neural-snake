extern crate rand;

use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use std::f32;

fn main() {
    let mut network = Network::new(&[2, 2, 1]);
    let mut dna_pool = DnaPool::new(20, 8);

    println!("{:?}", std::u32::MAX);
    loop {
        //let dna = Dna::new(8);

        network.update_weights(dna_pool.request());

        let mut results: Vec<f32> = Vec::new();
        results.push(network.run(vec![-1f32, -1f32])[0]);
        results.push(network.run(vec![ 1f32, -1f32])[0]);
        results.push(network.run(vec![-1f32,  1f32])[0]);
        results.push(network.run(vec![ 1f32,  1f32])[0]);

        let mut reward = 0f32;

        if results[0] < 0f32{
            reward += 1f32;
        }

        if results[1] > 0f32 {
            reward += 1f32;
        }

        if results[2] > 0f32 {
            reward += 1f32;
        }

        if results[3] < 0f32 {
            reward += 1f32;
        }


        let mut error = 0f32;
        error += (1f32 + results[0]).abs();
        error += (-1f32 + results[1]).abs();
        error += (-1f32 + results[2]).abs();
        error += (1f32 + results[3]).abs();
        

        if reward >= 4f32 {
            println!("{:?} {:?}", network.weights, results);
        }

        println!("{}", reward);

        dna_pool.evaluate(reward);
    }
}

struct DnaPool {
    pool: Vec<Dna>,
    index: usize,
    pool_size: usize,
    dna_size: usize
}

impl DnaPool {
    pub fn new(pool_size: usize, dna_size: usize) -> DnaPool {
        let mut pool: Vec<Dna> = Vec::new();

        for _ in 0..pool_size {
            pool.push(Dna::new(dna_size));
        }

        return DnaPool {
            pool: pool,
            index: 0,
            pool_size: pool_size,
            dna_size: dna_size
        }
    }

    pub fn request(&self) -> &Vec<f32> {
        let dna = &self.pool[self.index];
        return &dna.genomes;
    }

    pub fn evaluate(&mut self, fitness: f32) {
        self.pool[self.index].fitness = fitness;
        self.index += 1;

        if self.index >= self.pool_size {
            self.index = 0;
            self.evolve();
        }
    }

    fn pick_random(&self, sum: f32, rnd: f32) -> &Dna {
        let mut rnd = rnd;
        for dna in self.pool.iter() {
            if rnd < dna.fitness {
                return dna;
            }
            rnd -= dna.fitness;
        }
        panic!("Weighted random generator error");
    }

    fn normalize_weights(&mut self) {
        let mut max = 1f32;
        for dna in self.pool.iter() {
            if dna.fitness > max {
                max = dna.fitness;
            }
        }

        for dna in self.pool.iter_mut() {
            dna.fitness /= max;
        }
    }

    fn evolve(&mut self) {
        //self.normalize_weights();
        let sum = self.pool.iter().fold(0f32, |sum, dna| sum + dna.fitness);
        let mut rng = rand::thread_rng();
        let range = Range::new(0f32, sum);
        let mut new_pool: Vec<Dna> = Vec::with_capacity(self.pool_size);

        for _ in 0..self.pool_size {
            let a = self.pick_random(sum, range.ind_sample(&mut rng));
            let b = self.pick_random(sum, range.ind_sample(&mut rng));

            let mut child = a.crossover(b);
            child.mutate();
            new_pool.push(child);
        }

        self.pool = new_pool;    
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

    pub fn run(&self, input: Vec<f32>) -> Vec<f32> {
        let mut result = input;

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

    pub fn update_weights(&mut self, weights: &Vec<f32>) {
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