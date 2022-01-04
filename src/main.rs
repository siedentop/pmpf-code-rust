use rand::{distributions::Uniform, Rng};
use std::{
    collections::{HashSet, VecDeque},
    time::{self, Instant},
};
use timeit::timeit_loops;

/** Algorithms for converting 2D coordinates to and from the Hilbert index.

Here the Hilbert curve has been scaled and discretized, so that the
range {0, 1, ..., n^2 - 1} is mapped to coordinates
{0, 1, ..., n-1} x {0, 1, ..., n-1}. In the classical Hilbert curve,
the continuous interval [0,1] is mapped to the unit square [0,1]^2.
*/

type Coordinates = (usize, usize);
type Matrix = Vec<Vec<i32>>; // TODO: highly non-optimal
type Vector = Vec<i32>;

fn main() {
    let mut rng = rand::thread_rng();

    let range = Uniform::new(1, 11);
    let n: usize = 2usize.pow(11); // I observed pessimization with '14'.

    let start = time::Instant::now();
    // A = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    #[allow(non_snake_case)]
    let A = make_matrix(n, 1, 11);
    // v = [random.randint(1, 10) for _ in range(n)]
    let v: Vec<_> = (0..n).map(|_| rng.sample(&range)).collect();
    assert_eq!(v.len(), n);
    let mut output1: Vector = vec![0; n];
    let mut output2: Vector = vec![0; n];
    let end = time::Instant::now();
    println!("Initial data generation: {}s", (end - start).as_secs_f32());

    // Naive
    let timeit_count = 20;
    let total_n_seconds = timeit_loops! {timeit_count,
        {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
    };

    // reorder data
    let start = Instant::now();

    #[allow(non_snake_case)]
    let mut flattened_A = vec![0; n * n];
    let depth: usize = log2(n);
    let coordinate_iter: Vec<(usize, Coordinates)> =
        hilbert_iter(depth).into_iter().enumerate().collect();
    for (t, (i, j)) in &coordinate_iter {
        flattened_A[*t] = A[*i][*j];
    }

    let end = Instant::now();
    println!(
        "hilbert data preprocessing: {}s",
        (end - start).as_secs_f32()
    );

    // Hilbert Product
    let total_h_seconds = timeit_loops! {timeit_count,
        {hilbert_matrix_vector_product(&flattened_A,&v, &mut output2, &coordinate_iter);}
    };

    assert_eq!(output1, output2);

    // Print timings
    println!(
        "Naive: {}s ({}s per)",
        total_n_seconds,
        total_n_seconds / (timeit_count as f64)
    );

    println!(
        "Hilbert: {}s ({} s per)",
        total_h_seconds,
        total_h_seconds / (timeit_count as f64)
    );
    println!(
        "Improvement: {}%",
        100. * (1.0 - (total_h_seconds / total_n_seconds))
    );
}

fn log2(n: usize) -> usize {
    (n as f64).log2().floor() as usize
}

/// Create a matrix.
/// note that the representation Vec of Vec is not optimal.
fn make_matrix(n: usize, low: i32, high: i32) -> Matrix {
    let mut rng = rand::thread_rng(); // TODO: seed!
    let range = Uniform::new(low, high);
    (0..n)
        .map(|_| (0..n).map(|_| rng.sample(&range)).collect())
        .collect()
}

/// Naive product
#[allow(non_snake_case)]
fn naive_matrix_vector_product(A: &Matrix, v: &Vector, output: &mut Vector, n: usize) {
    // TODO: put asserts here to make sure no bounds checking happens.
    for i in 0..n {
        for j in 0..n {
            output[i] += A[i][j] * v[j];
        }
    }
}

#[allow(non_snake_case)]
fn hilbert_matrix_vector_product(
    flattened_A: &Vector,
    v: &Vector,
    output: &mut Vector,
    coordinate_iter: &Vec<(usize, Coordinates)>,
) {
    for (t, (i, j)) in coordinate_iter {
        output[*i] += flattened_A[*t] * v[*j];
    }
}

fn hilbert_iter(depth: usize) -> Vec<Coordinates> {
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
