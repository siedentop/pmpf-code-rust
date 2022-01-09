use jeremy_kun_math_rust::{naive_matrix_vector_product, setup_inputs};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rust_macos_perf::PerformanceCounters;

/// Experiment to compare Performance Counter with runtime.
/// Is PerfCounter a linear regression for runtime?
///

// use time::Timespec;

#[derive(Debug, Clone)]
struct Measurement {
    timing_s: f64,
    pc: PerformanceCounters,
    label: String,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    rust_macos_perf::init()?;

    let mut rng = ChaCha8Rng::seed_from_u64(10);

    for n in (5..14).map(|n| 2usize.pow(n)) {
        #[allow(non_snake_case)]
        let (A, v) = setup_inputs(n, &mut rng);

        let mut output1 = vec![0; n];

        let timeit_count = 20;
        let total_n_seconds = timeit::timeit_loops! {timeit_count,
            {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
        };

        let pc_naive = rust_macos_perf::timeit_loops! {timeit_count,
            {  naive_matrix_vector_product(&A, &v, &mut output1, n); }
        }?;
        println!(
            "{}, {}, {}, {}, {}, {}",
            n,
            total_n_seconds,
            pc_naive.cycles,
            pc_naive.branches,
            pc_naive.missed_branches,
            pc_naive.instructions
        );
    }
    Ok(())
}
