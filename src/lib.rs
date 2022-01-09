/** Algorithms for converting 2D coordinates to and from the Hilbert index.

Here the Hilbert curve has been scaled and discretized, so that the
range {0, 1, ..., n^2 - 1} is mapped to coordinates
{0, 1, ..., n-1} x {0, 1, ..., n-1}. In the classical Hilbert curve,
the continuous interval [0,1] is mapped to the unit square [0,1]^2.
*/
use rand::{distributions::Uniform, Rng};
use std::collections::{HashSet, VecDeque};

pub type Coordinates = (usize, usize);
type Matrix = Vec<i32>;
pub type Vector = Vec<i32>;

#[inline]
pub fn log2(n: usize) -> usize {
    (n as f64).log2().floor() as usize
}

/// Create a matrix.
/// note that the representation Vec of Vec is not optimal.
pub fn make_matrix(n: usize, low: i32, high: i32) -> Matrix {
    let mut rng = rand::thread_rng(); // TODO: seed!
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
pub fn flatten_matrix(
    coordinate_iter: &Vec<(usize, (usize, usize))>,
    A: Vec<i32>,
    n: usize,
) -> Vector {
    let mut flattened_A = vec![0; n * n];
    for (t, (i, j)) in coordinate_iter {
        flattened_A[*t] = A[flat_index(*i, *j, n)];
    }
    flattened_A
}

#[allow(non_snake_case)]
pub fn hilbert_matrix_vector_product(
    flattened_A: &Vector,
    v: &Vector,
    output: &mut Vector,
    coordinate_iter: &Vec<(usize, Coordinates)>,
) {
    for (t, (i, j)) in coordinate_iter {
        output[*i] += flattened_A[*t] * v[*j];
    }
}

pub fn hilbert_iter(depth: usize) -> Vec<Coordinates> {
    let mut result = Vec::new();
    let mut queue = VecDeque::from([('H', depth)]);

    let (mut i, mut j) = (0, 0);
    let non_terminals: HashSet<char> = "HABC".chars().collect();
    result.push((i, j));

    while let Some((symbol, depth)) = queue.pop_front() {
        if depth == 0 && !non_terminals.contains(&symbol) {
            match symbol {
                '↑' => {
                    i += 1;
                }
                '↓' => {
                    i -= 1;
                }
                '→' => {
                    j += 1;
                }
                '←' => {
                    j -= 1;
                }
                c => {
                    panic!("Unexpected symbol: {}", c);
                }
            }
            result.push((i, j));
        }
        if depth > 0 {
            match symbol {
                'H' => {
                    queue.push_back(('A', depth - 1));
                    queue.push_back(('↑', depth - 1));
                    queue.push_back(('H', depth - 1));
                    queue.push_back(('→', depth - 1));
                    queue.push_back(('H', depth - 1));
                    queue.push_back(('↓', depth - 1));
                    queue.push_back(('B', depth - 1));
                }
                'A' => {
                    queue.push_back(('H', depth - 1));
                    queue.push_back(('→', depth - 1));
                    queue.push_back(('A', depth - 1));
                    queue.push_back(('↑', depth - 1));
                    queue.push_back(('A', depth - 1));
                    queue.push_back(('←', depth - 1));
                    queue.push_back(('C', depth - 1));
                }
                'B' => {
                    queue.push_back(('C', depth - 1));
                    queue.push_back(('←', depth - 1));
                    queue.push_back(('B', depth - 1));
                    queue.push_back(('↓', depth - 1));
                    queue.push_back(('B', depth - 1));
                    queue.push_back(('→', depth - 1));
                    queue.push_back(('H', depth - 1));
                }
                'C' => {
                    queue.push_back(('B', depth - 1));
                    queue.push_back(('↓', depth - 1));
                    queue.push_back(('C', depth - 1));
                    queue.push_back(('←', depth - 1));
                    queue.push_back(('C', depth - 1));
                    queue.push_back(('↑', depth - 1));
                    queue.push_back(('A', depth - 1));
                }
                _ => {
                    // # terminal up/down/left/right symbols
                    // # must be preserved until the end
                    queue.push_back((symbol, depth - 1));
                }
            };
        }
    }
    return result;
}
