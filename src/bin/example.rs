use jeremy_kun_math_rust::{
    flat_index, hilbert_iter, hilbert_matrix_vector_product, log2, make_matrix,
    naive_matrix_vector_product, Coordinates, Vector,
};
/// The original example from Jeremy Kun's Python code.
use rand::{distributions::Uniform, Rng};
use rust_macos_perf::compare_perf_counters;
use std::time::{self, Instant};
use timeit::timeit_loops;

fn main() -> eyre::Result<()> {
    let mut rng = rand::thread_rng();

    rust_macos_perf::init()?;

    let range = Uniform::new(1, 11);
    let n: usize = 2usize.pow(11); // I observed a slowdown for the Hilbert code with '2^14'.

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

    let pc_naive = rust_macos_perf::timeit_loops! {timeit_count,
        {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
    }?;

    // reorder data
    let start = Instant::now();

    #[allow(non_snake_case)]
    let mut flattened_A = vec![0; n * n];
    let depth: usize = log2(n);
    let coordinate_iter: Vec<(usize, Coordinates)> =
        hilbert_iter(depth).into_iter().enumerate().collect();
    for (t, (i, j)) in &coordinate_iter {
        flattened_A[*t] = A[flat_index(*i, *j, n)];
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

    let pc_hilbert = rust_macos_perf::timeit_loops! {timeit_count,
        {  hilbert_matrix_vector_product(&flattened_A, &v, &mut output2, &coordinate_iter); }
    }?;

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

    println!("Naive: {:?}", pc_naive);
    println!("Hilbert: {:?}", pc_hilbert);
    println!(
        "Comparison: {}",
        compare_perf_counters(&pc_naive, &pc_hilbert)
    );

    Ok(())
}
