#[allow(non_snake_case)]
use jeremy_kun_math_rust::{
    hilbert_matrix_vector_product, naive_matrix_vector_product, setup_hilbert, setup_inputs, Vector,
};
use jeremy_kun_math_rust::{hilbert_matrix_vector_product_iter, log2};
#[cfg(feature = "macos-perf")]
use macos_perf::{compare_perf_counters, PerformanceCounters};
/// The original example from Jeremy Kun's Python code.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::{self, Instant};
use timeit::timeit_loops;

fn main() -> eyre::Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    #[cfg(feature = "macos-perf")]
    macos_perf::init()?;

    let n: usize = 2usize.pow(11); // I observed a slowdown for the Hilbert code with '2^14'.

    let start = time::Instant::now();

    #[allow(non_snake_case)]
    let (A, v) = setup_inputs(n, &mut rng);

    let mut output1: Vector = vec![0; n];
    let mut output2: Vector = vec![0; n];
    let mut output3: Vector = vec![0; n];
    let end = time::Instant::now();
    println!("Initial data generation: {}s", (end - start).as_secs_f32());

    // Naive
    let timeit_count = 20;
    let total_n_seconds = timeit_loops! {timeit_count,
        {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
    };

    #[cfg(feature = "macos-perf")]
    let pc_naive = macos_perf::timeit_loops! {timeit_count,
        {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
    }?;

    // reorder data
    let start = Instant::now();

    #[allow(non_snake_case)]
    let (hilbert_iter, flattened_A) = setup_hilbert(n, A);

    let end = Instant::now();
    println!(
        "hilbert data preprocessing: {}s",
        (end - start).as_secs_f32()
    );

    // Hilbert Product
    let total_h_seconds = timeit_loops! {timeit_count,
        {hilbert_matrix_vector_product(&flattened_A,&v, &mut output2, &hilbert_iter);}
    };

    #[cfg(feature = "macos-perf")]
    let pc_hilbert = macos_perf::timeit_loops! {timeit_count,
        {  hilbert_matrix_vector_product(&flattened_A, &v, &mut output2, &hilbert_iter); }
    }?;

    assert_eq!(output1, output2);

    // Hilbert Product Iterator
    let depth = log2(n);
    let total_hilbert_iter_seconds = timeit_loops! {timeit_count,
        {hilbert_matrix_vector_product_iter(&flattened_A, &v, &mut output3, depth);}
    };

    #[cfg(feature = "macos-perf")]
    let pc_hilbert_iter = macos_perf::timeit_loops! {timeit_count,
        {  hilbert_matrix_vector_product_iter(&flattened_A, &v, &mut output3, depth); }
    }?;
    assert_eq!(output1, output3);

    print_timings(
        total_n_seconds,
        total_h_seconds,
        total_hilbert_iter_seconds,
        timeit_count as f64,
    );

    #[cfg(feature = "macos-perf")]
    print_perf_counters(pc_naive, pc_hilbert, pc_hilbert_iter);
    Ok(())
}

fn print_timings(
    total_n_seconds: f64,
    total_h_seconds: f64,
    total_hilbert_iter_seconds: f64,
    timeit_count: f64,
) {
    println!(
        "Naive: {}s ({}s per)",
        total_n_seconds,
        total_n_seconds / timeit_count
    );
    println!(
        "Hilbert: {:+e}s ({:+e} s per)",
        total_h_seconds,
        total_h_seconds / timeit_count
    );
    println!(
        "Hilbert (iter): {:+e}s ({:+e} s per)",
        total_hilbert_iter_seconds,
        total_hilbert_iter_seconds / timeit_count
    );
    println!(
        "Improvement: {}% {}%",
        100. * (1.0 - (total_h_seconds / total_n_seconds)),
        100. * (1.0 - (total_hilbert_iter_seconds / total_n_seconds))
    );
}

/// Print performance counters.
#[cfg(feature = "macos-perf")]
fn print_perf_counters(
    pc_naive: PerformanceCounters,
    pc_hilbert: PerformanceCounters,
    pc_hilbert_iter: PerformanceCounters,
) {
    println!("Naive: {:?}", pc_naive);
    println!("Hilbert: {:?}", pc_hilbert);
    println!("Hilbert (iter): {:?}", pc_hilbert_iter);
    println!(
        "Comparison: {}",
        compare_perf_counters(&pc_naive, &pc_hilbert)
    );
    println!(
        "Comparison (iter): {}",
        compare_perf_counters(&pc_naive, &pc_hilbert_iter)
    );
}
