# About this code

This repository contains code for matrix multiplication using Hilbert Curves.

The original code ([2]) is part of @j2kun's code for his next book [3]. It was written in Python -- but
Python did not show the expected performance benefit of using the Hilbert Curve for matrix multiplication
(as compared to a naive multiplication).

Thus I decided to give it a go in a compiled language.

[1]: https://github.com/j2kun/pmfp-code/pull/21/commits/312150eb26d4c29a70c71b69478a59f985f11f81
[2]: https://github.com/j2kun/pmfp-code/pull/21
[3]: https://github.com/j2kun/pmfp-code/

# How to run

1. Install a Rust toolchain from here: https://rustup.rs
2. Run `cargo run --release`
3. Observe the following output:

```shell
Initial data generation: 0.061384752s
hilbert data preprocessing: 0.21851613s
Naive: 0.0115141s (0.0005757049999999999s per)
Hilbert: 0.0090037s (0.000450185 s per)
Improvement: 21.802833048175707%
```

(Tested on my M1 Macbook Air, RustC 1.56.0)

## Apple Silicon devices

There is a feature flag, called `macos-perf`, which provides more details on macOS M1-based computers.

Run: `sudo cargo run --features macos-perf --release --quiet --bin example`
to see more detailed results.
