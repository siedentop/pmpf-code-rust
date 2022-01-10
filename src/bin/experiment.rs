use jeremy_kun_math_rust::{
    hilbert_matrix_vector_product, naive_matrix_vector_product, setup_hilbert, setup_inputs,
};
#[cfg(feature = "macos-perf")]
use macos_perf::PerformanceCounters;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Experiment to compare Performance Counter with runtime.
/// Is PerfCounter a linear regression for runtime?
///

// use time::Timespec;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    #[cfg(feature = "macos-perf")]
    macos_perf::init()?;

    let mut rng = ChaCha8Rng::seed_from_u64(10);
    let timeit_count = 20;

    let matrix_sizes = (5..14).map(|n| 2usize.pow(n)).collect::<Vec<_>>();

    for n in matrix_sizes.iter() {
        #[allow(non_snake_case)]
        let (A, v) = setup_inputs(*n, &mut rng);
        let mut output1 = vec![0; *n];

        let total_n_seconds = timeit::timeit_loops! {timeit_count,
            {  naive_matrix_vector_product(&A, &v, &mut output1, *n); }
        };

        #[cfg(feature = "macos-perf")]
        let pc_naive = macos_perf::timeit_loops! {timeit_count,
            {  naive_matrix_vector_product(&A, &v, &mut output1, *n); }
        }?;
        #[cfg(feature = "macos-perf")]
        print_row("naive", *n, total_n_seconds, pc_naive);
        #[cfg(not(feature = "macos-perf"))]
        println!("naive, {}, {}", *n, total_n_seconds);
    }

    // Re-seed RNG.
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    for n in matrix_sizes {
        #[allow(non_snake_case)]
        let (A, v) = setup_inputs(n, &mut rng);
        #[allow(non_snake_case)]
        let (coordinate_iter, flattened_A) = setup_hilbert(n, A);
        let mut output = vec![0; n];

        // Hilbert Product
        let total_h_seconds = timeit::timeit_loops! {timeit_count,
            {hilbert_matrix_vector_product(&flattened_A,&v, &mut output, &coordinate_iter);}
        };

        #[cfg(feature = "macos-perf")]
        let pc_hilbert = macos_perf::timeit_loops! {timeit_count,
            {  hilbert_matrix_vector_product(&flattened_A, &v, &mut output, &coordinate_iter); }
        }?;
        #[cfg(feature = "macos-perf")]
        print_row("hilbert", n, total_h_seconds, pc_hilbert);
        #[cfg(not(feature = "macos-perf"))]
        println!("hilbert, {}, {}", n, total_h_seconds);
    }
    Ok(())
}

#[cfg(feature = "macos-perf")]
fn print_row(label: &str, n: usize, total_n_seconds: f64, pc_naive: PerformanceCounters) {
    println!(
        "{}, {}, {}, {}, {}, {}, {}",
        label,
        n,
        total_n_seconds,
        pc_naive.cycles,
        pc_naive.branches,
        pc_naive.missed_branches,
        pc_naive.instructions
    );
}
