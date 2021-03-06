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
use std::f32;

fn main() {
    let mut network = Network::new(&[4, 7, 7, 7, 3]);
    let mut dna_pool = DnaPool::new(50, network.weight_count);
    network.update_weights(dna_pool.request());

    let board = Board::new(20, 20);
    let mut snake = Snake::new(board);

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

    let mut speed = 1;
    while let Some(e) = window.next() {
        if let Some(_) = e.update_args() {
            for _ in 0..speed {
                //println!("{}", snake.direction);
                let inputs = snake.get_inputs();
                let outputs = network.run(inputs);
                snake.apply_outputs(outputs);
                snake.update();
                //println!("{}", snake.direction);

                if !snake.alive {
                    dna_pool.evaluate(snake.parts.len() as f32);
                    network.update_weights(dna_pool.request());
                    snake.reset();
                }
            }
        }

        if let Some(_) = e.render_args() {
            for x in 0..snake.board.width {
                for y in 0..snake.board.height {
                    canvas.put_pixel(x as u32, y as u32, Rgba([255, 255, 255, 255]));
                }
            }

            canvas.put_pixel(
                snake.board.food.x as u32,
                snake.board.food.y as u32,
                Rgba([255, 0, 0, 255]),
            );

            for point in snake.parts.iter() {
                canvas.put_pixel(point.x as u32, point.y as u32, Rgba([0, 0, 0, 255]));
            }

            texture.update(&mut window.encoder, &canvas).unwrap();

            window.draw_2d(&e, |c, g| {
                clear([0.0; 4], g);
                image(&texture, c.transform.scale(30f64, 30f64), g);
            });
        }

        e.mouse_scroll(|_, y| {
            if y > 0f64 {
                speed += 100;
            } else if speed > 1 {
                speed -= 100;
            }

            println!("{}", speed);
        });
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
        return Point::new(
            rangeX.ind_sample(&mut rng) as isize,
            rangeY.ind_sample(&mut rng) as isize,
        );
    }
}

struct Snake {
    direction: usize,
    parts: VecDeque<Point>,
    board: Board,
    alive: bool,
    steps_without_food: usize,
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
            steps_without_food: 0,
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

    fn touching(&self, x: isize, y: isize) -> bool {
        for part in self.parts.iter() {
            if part.x == x && part.y == y  {
                return true;
            }
        }
        return false;
    }

    fn check_collisions(&mut self) {
        if !self.in_bounds() || self.self_collision() {
            self.alive = false;
            return;
        }
    }

    fn get_food_input(&self) -> f32 {
        let head = self.parts.front().unwrap();
        let food = &self.board.food;
        let x = food.x - head.x;
        let y = food.y - head.y;
        let vector = (y as f32).atan2(x as f32);

        let direction = match self.direction {
            0 => 0f32,
            1 => -f32::consts::PI / 2f32,
            2 => -f32::consts::PI,
            3 => f32::consts::PI / 2f32,
            _ => panic!("Invalid direction"),
        };

        let mut sum = vector + direction;

        if sum > f32::consts::PI {
            sum -= -f32::consts::PI;
        } else if sum < -f32::consts::PI {
            sum += f32::consts::PI;
        }

        return sum / f32::consts::PI;
    }

    fn peek_at(&self, direction: usize) -> f32 {
        const DIRECTIONS: [[isize; 2]; 4] = [[1, 0], [0, 1], [-1, 0], [0, -1]];
        let head = self.parts.front().unwrap();
        let mut i = 1;
        loop {
            let x = head.x + DIRECTIONS[direction][0] * i;
            let y = head.y + DIRECTIONS[direction][1] * i;
            if !self.board.in_bounds(x, y) || self.touching(x, y) {
                // FIXME: This works only if board is square
                return 1f32 - ((i - 1) as f32 / (self.board.width as f32 - 0.5f32));
            }
            i += 1;
        }
    }

    fn get_distance_inputs(&self) -> Vec<f32> {
        let front = self.peek_at(self.direction);
        let left = self.peek_at((self.direction + 3) % 4);
        let right = self.peek_at((self.direction + 1) % 4);

        return vec![front, 0f32, 0f32];
    }

    pub fn get_inputs(&self) -> Vec<f32> {
        let mut inputs = Vec::with_capacity(4);

        inputs.append(&mut self.get_distance_inputs());
        inputs.push(self.get_food_input());

        return inputs;
    }

    pub fn apply_outputs(&mut self, outputs: Vec<f32>) {
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
            1 => {}
            2 => self.direction += 1,
            _ => panic!("Invalid direction index"),
        }

        self.direction %= 4;
    }

    pub fn update(&mut self) {
        if !self.alive {
            return;
        };

        let mut new_head = self.parts.front().unwrap().clone();

        match self.direction {
            0 => new_head.x += 1,
            1 => new_head.y += 1,
            2 => new_head.x -= 1,
            3 => new_head.y -= 1,
            _ => panic!("Invalid snake direction"),
        }

        self.parts.push_front(new_head);

        self.steps_without_food += 1;

        if *self.parts.front().unwrap() != self.board.food {
            self.parts.pop_back();
        } else {
            self.board.regenerate_food(&self.parts);
            self.steps_without_food = 0;
        }

        if self.steps_without_food > 500 {
            self.alive = false;
            return;
        }

        self.check_collisions();
    }

    pub fn reset(&mut self) {
        self.alive = true;
        self.parts = VecDeque::new();
        self.parts.push_front(Point::new(9, 9));
        self.direction = 0;
        self.steps_without_food = 0;
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
            food: Point::random(width, height),
        };
    }

    pub fn in_bounds(&self, x: isize, y: isize) -> bool {
        return x >= 0 && y >= 0 && x < self.width as isize && y < self.height as isize;
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
    best: Dna,
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


        println!(
            "Generation evolution, avg fitness: {}",
            sum / self.pool.len() as f32
        );
        let mut rng = rand::thread_rng();
        let range = Range::new(0f32, sum);
        let mut new_pool: Vec<Dna> = Vec::with_capacity(self.pool_size);

        for _ in 0..self.pool_size-1 {
            let a = self.pick_random(sum, range.ind_sample(&mut rng));
            let b = self.pick_random(sum, range.ind_sample(&mut rng));

            let mut child = a.crossover(b);
            child.mutate();
            new_pool.push(child);
        }

        new_pool.push(self.best.clone());

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
        let range = Range::new(0, 550);
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
