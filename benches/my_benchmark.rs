use iai::black_box;
use jeremy_kun_math_rust::{
    hilbert_matrix_vector_product, naive_matrix_vector_product, setup_hilbert, setup_inputs, Vector,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_naive() {
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    let n: usize = 2usize.pow(11); // I observed a slowdown for the Hilbert code with '2^14'.

    #[allow(non_snake_case)]
    let (A, v) = setup_inputs(n, &mut rng);

    let mut output1: Vector = vec![0; n];

    for _ in 0..10 {
        naive_matrix_vector_product(&A, &v, &mut output1, n);
    }
}

fn bench_hilbert() {
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    let n: usize = 2usize.pow(11); // I observed a slowdown for the Hilbert code with '2^14'.

    #[allow(non_snake_case)]
    let (A, v) = setup_inputs(n, &mut rng);

    let mut output1: Vector = vec![0; n];

    #[allow(non_snake_case)]
    let (hilbert_iter, flattened_A) = setup_hilbert(n, A);

    for _ in 0..10 {
        hilbert_matrix_vector_product(&flattened_A, black_box(&v), &mut output1, &hilbert_iter);
    }
}

iai::main!(bench_naive, bench_hilbert);
