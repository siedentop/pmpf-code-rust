/** Algorithms for converting 2D coordinates to and from the Hilbert index.

Here the Hilbert curve has been scaled and discretized, so that the
range {0, 1, ..., n^2 - 1} is mapped to coordinates
{0, 1, ..., n-1} x {0, 1, ..., n-1}. In the classical Hilbert curve,
the continuous interval [0,1] is mapped to the unit square [0,1]^2.
*/
use rand::{distributions::Uniform, Rng};
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

pub type Coordinates = (usize, usize);
type Matrix = Vec<i32>;
pub type Vector = Vec<i32>;

#[inline]
pub fn log2(n: usize) -> usize {
    (n as f64).log2().floor() as usize
}

/// Create a matrix.
/// note that the representation Vec of Vec is not optimal.
pub fn make_matrix<R: rand::Rng>(n: usize, low: i32, high: i32, rng: &mut R) -> Matrix {
    let range = Uniform::new(low, high);
    (0..(n * n)).map(|_| rng.sample(&range)).collect()
}

/// Naive product
#[allow(non_snake_case)]
pub fn naive_matrix_vector_product(A: &Matrix, v: &Vector, output: &mut Vector, n: usize) {
    // // TODO: put asserts here to make sure no bounds checking happens.
    // assert_eq!(output.len(), n);
    // assert_eq!(A.len(), n * n);
    // assert_eq!(v.len(), n);
    for i in 0..n {
        for j in 0..n {
            output[i] += A[flat_index(i, j, n)] * v[j];
        }
    }
}

/// Converts [i][j] into [n*i+j]
#[inline]
fn flat_index(i: usize, j: usize, n: usize) -> usize {
    n * i + j
}

/// Flatten matrix A according to the provided Hilbert coordinates.
#[allow(non_snake_case)]
pub fn flatten_matrix(depth: usize, A: Vec<i32>, n: usize) -> Vector {
    let mut flattened_A = vec![0; n * n];
    for (t, (i, j)) in HilbertIter::new(depth) {
        flattened_A[t] = A[flat_index(i, j, n)];
    }
    flattened_A
}

#[allow(non_snake_case)]
pub fn hilbert_matrix_vector_product(
    flattened_A: &Vector,
    v: &Vector,
    output: &mut Vector,
    hilbert_iter: &Vec<(usize, Coordinates)>,
) {
    for (t, (i, j)) in hilbert_iter {
        output[*i] += flattened_A[*t] * v[*j];
    }
}

/// `hilbert_matrix_vector_product` but Hilbert index is an iterator.
#[allow(non_snake_case)]
pub fn hilbert_matrix_vector_product_iter(
    flattened_A: &Vector,
    v: &Vector,
    output: &mut Vector,
    depth: usize,
) {
    for (t, (i, j)) in HilbertIter::new(depth) {
        output[i] += flattened_A[t] * v[j];
    }
}

struct HilbertIter {
    /// Number of steps remaining
    n: usize,
    index: usize,
    i: usize,
    j: usize,
    queue: VecDeque<(char, usize)>,
    buffer: Option<(usize, Coordinates)>,
}

impl HilbertIter {
    pub fn new(depth: usize) -> Self {
        let n = 2usize.pow(depth as u32);
        let n = n * n + 1;
        let queue = VecDeque::from([('H', depth)]);
        Self {
            n,
            index: 1,
            i: 0,
            j: 0,
            queue,
            buffer: Some((0, (0, 0))),
        }
    }

    fn step(&mut self) {
        while self.buffer.is_none() && !self.queue.is_empty() {
            let (symbol, depth) = self.queue.pop_front().unwrap();
            if depth == 0 {
                let non_terminal = match symbol {
                    '???' => {
                        self.i += 1;
                        true
                    }
                    '???' => {
                        self.i -= 1;
                        true
                    }
                    '???' => {
                        self.j += 1;
                        true
                    }
                    '???' => {
                        self.j -= 1;
                        true
                    }
                    _ => false,
                };
                if non_terminal {
                    self.buffer = Some((self.index, (self.i, self.j)));
                    self.index += 1;
                }
            }
            if depth > 0 {
                match symbol {
                    'H' => {
                        self.queue.push_back(('A', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('H', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('H', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('B', depth - 1));
                    }
                    'A' => {
                        self.queue.push_back(('H', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('A', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('A', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('C', depth - 1));
                    }
                    'B' => {
                        self.queue.push_back(('C', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('B', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('B', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('H', depth - 1));
                    }
                    'C' => {
                        self.queue.push_back(('B', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('C', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('C', depth - 1));
                        self.queue.push_back(('???', depth - 1));
                        self.queue.push_back(('A', depth - 1));
                    }
                    _ => {
                        // # terminal up/down/left/right symbols
                        // # must be preserved until the end
                        self.queue.push_back((symbol, depth - 1));
                    }
                };
            }
        }
    }
}

impl Iterator for HilbertIter {
    type Item = (usize, Coordinates);

    fn next(&mut self) -> Option<Self::Item> {
        if self.n > 0 {
            self.n -= 1;
        } else {
            panic!("Hilbert iteration is out of bounds");
        }
        self.step();
        self.buffer.take()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.n, Some(self.n))
    }
}

/// Generate (A, v) as inputs for matrix multiplication
pub fn setup_inputs(n: usize, rng: &mut ChaCha8Rng) -> (Vec<i32>, Vec<i32>) {
    let range = Uniform::new(1, 11);

    #[allow(non_snake_case)]
    let A = make_matrix(n, 1, 11, rng);
    let v: Vec<_> = (0..n).map(|_| rng.sample(&range)).collect();
    assert_eq!(v.len(), n);
    (A, v)
}

/// Setup (coordinates, flattened_A) for Hilbert multiplication
#[allow(non_snake_case)]
pub fn setup_hilbert(n: usize, A: Vec<i32>) -> (Vec<(usize, Coordinates)>, Vec<i32>) {
    assert_eq!(n * n, A.len());
    let depth: usize = log2(n);
    let hilbert_iter: Vec<_> = HilbertIter::new(depth).collect();
    println!("Hilbert matrix size: {}", hilbert_iter.len());

    #[allow(non_snake_case)]
    let flattened_A = flatten_matrix(depth, A, n);
    (hilbert_iter, flattened_A)
}
/// Setup (depth, flattened_A) for Hilbert multiplication
#[allow(non_snake_case)]
pub fn setup_hilbert_iter(n: usize, A: Vec<i32>) -> (usize, Vec<i32>) {
    assert_eq!(n * n, A.len());
    let depth: usize = log2(n);

    #[allow(non_snake_case)]
    let flattened_A = flatten_matrix(depth, A, n);
    (depth, flattened_A)
}

#[cfg(test)]
mod test {
    use std::time::{self};

    use insta::assert_yaml_snapshot;
    use rand::distributions::Uniform;
    use rand::Rng;
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
    use timeit::timeit_loops;

    use crate::{
        hilbert_matrix_vector_product, hilbert_matrix_vector_product_iter, make_matrix,
        naive_matrix_vector_product,
    };

    #[test]
    fn test_equality_of_implementation() {
        let mut rng = ChaCha8Rng::seed_from_u64(10);

        let range = Uniform::new(1, 11);
        let n: usize = 2usize.pow(8);

        let start = time::Instant::now();
        // A = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
        #[allow(non_snake_case)]
        let A = make_matrix(n, 1, 11, &mut rng);
        // v = [random.randint(1, 10) for _ in range(n)]
        let v: Vec<_> = (0..n).map(|_| rng.sample(&range)).collect();
        assert_eq!(v.len(), n);
        let mut output1 = vec![0; n];
        let mut output2 = vec![0; n];
        let mut output3 = vec![0; n];
        let end = time::Instant::now();
        println!("Initial data generation: {}s", (end - start).as_secs_f32());

        // Naive
        let timeit_count = 3;
        let _ = timeit_loops! {timeit_count,
            {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
        };

        // reorder data
        #[allow(non_snake_case)]
        let (depth, flattened_A) = super::setup_hilbert_iter(n, A.clone());

        // Hilbert Product
        let _ = timeit_loops! {timeit_count,
            {hilbert_matrix_vector_product_iter(&flattened_A,&v, &mut output2, depth);}
        };

        // non-iterative version
        let (hilbert_iter, flattened_a) = super::setup_hilbert(n, A);

        let _ = timeit_loops! {timeit_count,
            {hilbert_matrix_vector_product(&flattened_a,&v, &mut output3, &hilbert_iter);}
        };

        assert_eq!(output1, output2);
        assert_eq!(output1, output3);
        assert_yaml_snapshot!(output2);
    }
}
