extern crate image;
extern crate opengl_graphics;
extern crate piston;
extern crate piston_window;
extern crate rand;

use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use std::collections::VecDeque;
use piston_window::*;
use opengl_graphics::OpenGL;
use image::{ImageBuffer, Rgba};
use texture::Filter;

fn main() {
    let mut network = Network::new(&[4, 6, 6, 3]);
    let mut dna_pool = DnaPool::new(30, network.weight_count);
    network.update_weights(dna_pool.request());

    let board = Board::new(20, 20);
    let mut snake = Snake::new(board);

    println!("{:?}", snake.parts);

    let mut canvas = ImageBuffer::new(snake.board.width as u32, snake.board.height as u32);
    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow = WindowSettings::new("Neural Snake", [640, 480])
        .opengl(opengl)
        .build()
        .unwrap();
    let mut texture = Texture::from_image(
        &mut window.factory,
        &canvas,
        &TextureSettings::new().filter(Filter::Nearest),
    ).unwrap();

    let mut counter = 0;

    while let Some(e) = window.next() {
        if let Some(_) = e.update_args() {
            if counter % 30 == 0 {
                let inputs = snake.get_inputs();
                let outputs = network.run(inputs);
                snake.apply_outputs(outputs);
                snake.update();

                if !snake.alive {
                    dna_pool.evaluate(snake.parts.len() as f32);
                    network.update_weights(dna_pool.request());
                    snake.reset();
                }
            }
            
            counter += 1; 
        }

        if let Some(_) = e.render_args() {
            for x in 0..snake.board.width {
                for y in 0..snake.board.height {
                    canvas.put_pixel(x as u32, y as u32, Rgba([255, 255, 255, 255]));
                }
            }

            canvas.put_pixel(snake.board.food.x as u32, snake.board.food.y as u32, Rgba([255, 0, 0, 255]));

            for point in snake.parts.iter() {
                canvas.put_pixel(point.x as u32, point.y as u32, Rgba([0, 0, 0, 255]));
            }

            texture.update(&mut window.encoder, &canvas).unwrap();

            window.draw_2d(&e, |c, g| {
                clear([0.0; 4], g);
                image(&texture, c.transform.scale(30f64, 30f64), g);
            });
        }
    }
}

#[derive(Debug, Clone, Eq)]
struct Point {
    x: isize,
    y: isize,
}

impl PartialEq for Point {
    fn eq(&self, point: &Point) -> bool {
        return self.x == point.x && self.y == point.y;
    }
}

impl Point {
    pub fn new(x: isize, y: isize) -> Point {
        return Point { x: x, y: y };
    }

    pub fn random(max_x: usize, max_y: usize) -> Point {
        let rangeX = Range::new(0, max_x);
        let rangeY = Range::new(0, max_y);
        let mut rng = rand::thread_rng();
        return Point::new(rangeX.ind_sample(&mut rng) as isize, rangeY.ind_sample(&mut rng) as isize);
    }
}

struct Snake {
    direction: usize,
    parts: VecDeque<Point>,
    board: Board,
    alive: bool,
}

impl Snake {
    pub fn new(board: Board) -> Snake {
        let mut parts: VecDeque<Point> = VecDeque::new();
        parts.push_front(Point::new(9, 9));

        return Snake {
            direction: 0,
            parts: parts,
            board: board,
            alive: true,
        };
    }

    fn in_bounds(&self) -> bool {
        let head = self.parts.front().unwrap();
        return head.x >= 0 && head.y >= 0 && head.x < self.board.width as isize
            && head.y < self.board.height as isize;
    }

    fn self_collision(&self) -> bool {
        let head = self.parts.front().unwrap();

        for part in self.parts.iter().skip(1) {
            if part == head {
                return true;
            }
        }

        return false;
    }

    fn check_collisions(&mut self) {
        if !self.in_bounds() || self.self_collision() {
            self.alive = false;
            return
        }
    }

    fn get_inputs(&self) -> Vec<f32> {
        let mut inputs = Vec::with_capacity(4);
        inputs.push(0f32);
        inputs.push(0f32);
        inputs.push(0f32);
        inputs.push(0f32);
        return inputs;
    }

    fn apply_outputs(&mut self, outputs: Vec<f32>) {
        let mut max = -1f32;
        let mut max_index = 0;
        for (index, output) in outputs.iter().enumerate() {
            if *output > max {
                max = *output;
                max_index = index;
            }
        }

        match max_index {
            0 => self.direction += 3,
            1 => {},
            2 => self.direction += 1,
            _ => panic!("Invalid direction index")
        }

        self.direction %= 4;
    }

    pub fn update(&mut self) {
        if !self.alive {
            return;
        };

        let mut new_head = self.parts.front().unwrap().clone();

        match self.direction {
            0 => new_head.y -= 1,
            1 => new_head.x += 1,
            2 => new_head.y += 1,
            3 => new_head.x -= 1,
            _ => panic!("Invalid snake direction"),
        }

        self.parts.push_front(new_head);

        if *self.parts.front().unwrap() != self.board.food {
            self.parts.pop_back();
        } else {
            self.board.regenerate_food(&self.parts);
        }
        self.check_collisions();
    }

    pub fn reset(&mut self) {
        self.alive = true;
        self.parts = VecDeque::new();
        self.parts.push_front(Point::new(9, 9));
        self.direction = 0;
    }
}

struct Board {
    width: usize,
    height: usize,
    food: Point,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Board {
        return Board {
            width: width,
            height: height,
            food: Point::new(10, 10),
        };
    }

    pub fn regenerate_food(&mut self, snake_parts: &VecDeque<Point>) {
        loop {
            let point = Point::random(self.width, self.height);
            for part in snake_parts {
                if point == *part {
                    continue;
                }
            }
            self.food = point;
            return;
        }
    }
}

struct DnaPool {
    pool: Vec<Dna>,
    index: usize,
    pool_size: usize,
    best: Dna
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
            best: Dna::new(dna_size),
        };
    }

    pub fn request(&self) -> &Vec<f32> {
        let dna = &self.pool[self.index];
        return &dna.genomes;
    }

    pub fn evaluate(&mut self, fitness: f32) {
        self.pool[self.index].fitness = fitness;
        if fitness > self.best.fitness {
            self.best = self.pool[self.index].clone();
        }

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

#[derive(Debug, Clone)]
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
    weight_count: usize,
}

impl Network {
    pub fn new(sizes: &[usize]) -> Network {
        let mut weight_count = 0;
        let mut layers: Vec<Vec<Vec<f32>>> = Vec::new();
        for i in 0..(sizes.len() - 1) {
            let mut layer: Vec<Vec<f32>> = Vec::new();
            for _ in 0..sizes[i + 1] {
                let mut node: Vec<f32> = Vec::new();
                for _ in 0..sizes[i] {
                    node.push(0f32);
                    weight_count += 1;
                }

                if i != sizes.len() - 2 {
                    node.push(0f32);
                    weight_count += 1;
                }

                node.shrink_to_fit();
                layer.push(node);
            }
            layer.shrink_to_fit();
            layers.push(layer);
        }

        return Network {
            weights: layers,
            weight_count: weight_count,
        };
    }

    pub fn run(&self, input: Vec<f32>) -> Vec<f32> {
        let mut result = input;

        for (i, layer) in self.weights.iter().enumerate() {
            let mut current: Vec<f32> = Vec::new();

            for node in layer {
                let mut sum: f32 = 0.0;
                for (index, weight) in node.iter().enumerate() {
                    if index >= result.len() {
                        sum += weight;
                        break;
                    }
                    sum += result[index] * weight;
                }
                //println!("{}", sum);
                if i < self.weights.len() - 1 {
                    sum = tanh(sum);
                }

                current.push(sum);
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
