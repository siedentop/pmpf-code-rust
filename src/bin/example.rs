#[allow(non_snake_case)]
use jeremy_kun_math_rust::{
    hilbert_matrix_vector_product, naive_matrix_vector_product, setup_hilbert, setup_inputs, Vector,
};
/// The original example from Jeremy Kun's Python code.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rust_macos_perf::compare_perf_counters;
use std::time::{self, Instant};
use timeit::timeit_loops;

fn main() -> eyre::Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    rust_macos_perf::init()?;

    let n: usize = 2usize.pow(11); // I observed a slowdown for the Hilbert code with '2^14'.

    let start = time::Instant::now();

    #[allow(non_snake_case)]
    let (A, v) = setup_inputs(n, &mut rng);

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
    let (coordinate_iter, flattened_A) = setup_hilbert(n, A);

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
