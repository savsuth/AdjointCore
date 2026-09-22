# AdjointCore

AdjointCore is a reverse-mode Adjoint Algorithmic Differentiation (AAD)
engine, implemented from first principles in Rust. It prices a European Call
option by Monte Carlo simulation and, within the same computation, recovers
its Greeks (Delta, Vega, and Rho) from a single backward sweep per simulated
path. The engine is exposed to Python through PyO3 and `maturin`, allowing
it to be imported and called like any standard Python package once built.

No external autodiff crate is used anywhere in this project. The tape, the
Dual number, every operator overload, and the mathematical special
functions are all implemented in this repository.

## Architecture

- **Tape (`src/tape.rs`)**: A thread-local, arena-backed log of operations.
  Each `Node` stores up to two parent indices along with their local
  partial derivatives, packed into `u32`/`f64` fields rather than `usize`.
  This reduces the array scanned during the backward sweep by roughly 25%.
  The arena itself is a `bumpalo::Bump`, leaked to `'static` once per
  thread so the tape's `BumpVec` can borrow it without requiring a
  self-referential struct.

- **Pathwise differentiation**: Paths are differentiated pathwise
  (∇E[V] = E[∇V]). Each path records its own forward pass, performs its
  backward sweep, and is reset before the next path begins. This reset
  operation calls `nodes.clear()` rather than `Bump::reset()`, which
  preserves the vector's already-grown capacity instead of releasing it.
  As a result, from the second path onward, the forward pass performs zero
  heap allocations.

- **Dual number (`src/dual.rs`)**: A 16-byte `Copy` handle formatted as
  `{ value: f64, index: u32 }`, which holds no derivative data of its
  own — every partial resides on the tape. The `Gradient` struct reads
  directly from the tape's adjoint buffer rather than returning a copy,
  ensuring `.backward()` performs no allocations.

- **Operators (`src/ops.rs`)**: `Add`, `Sub`, `Mul`, and `Div` are
  implemented across all three relevant operand shapes (`Dual op Dual`,
  `Dual op f64`, `f64 op Dual`). The engine also supports `exp`, `sqrt`,
  `ln`, `norm_cdf`, and `max_zero` (the option payoff kink, whose
  subgradient is taken as 0 exactly at the kink itself).

- **Normal CDF (`src/special.rs`)**: The rational approximation commonly
  used across quantitative finance libraries is often described as double
  precision. Measured against 60-digit references, its accuracy turns out
  to be a little more modest than that label suggests — its tail alone
  carries an error of approximately 8.7e-9 near |x| = 8. This
  implementation takes a different approach, splitting the domain at
  |x| = 1.5. An all-positive-term `erf` series is used below that
  threshold, and a Mills-ratio continued fraction (evaluated by modified
  Lentz and run to full convergence rather than truncated early) is used
  above it. Measured against the same 60-digit references, the worst-case
  error is 2.6e-15, approximately six orders of magnitude tighter than the
  approximation it replaces.

- **Monte Carlo pricer (`src/pricer.rs`)**: An Euler-Maruyama simulation of
  geometric Brownian motion. Spot, Rate, and Volatility are recorded as
  tape variables, allowing Delta, Rho, and Vega to be recovered from the
  same sweep that produces the price. Strike and Time remain plain
  constants, since no Greek is required with respect to either. The
  standard error of each output quantity is tracked with a streaming
  accumulator (Welford's algorithm). The test suite verifies agreement
  with closed-form Black-Scholes to within five standard errors per
  quantity — a statistically grounded tolerance, rather than one fixed
  bound applied uniformly to all four estimators of differing variance.

- **PyO3 bridge (`src/python.rs`)**: A single function, `mc_pricer`,
  returns `{"price", "delta", "vega", "rho"}` as a Python dictionary. Once
  built, the engine is accessible via `import adjoint_core`.

## Building

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
python main.py
```

## Testing

```bash
cargo test
```

The suite comprises 32 tests across four modules: the tape's accumulation
and reset behavior, the normal CDF kernel verified against independently
generated reference values, every operator and math function verified
against its analytical derivative, and the Monte Carlo engine verified
against closed-form Black-Scholes with an additional cross-seed consistency
check.

## Benchmarking

```bash
python benchmark.py
```

This script reports two figures: per-call latency (a single `mc_pricer`
invocation covering all simulated paths) and per-path latency (the
per-call figure divided across those paths). Because the two figures
answer different performance questions, both are reported.
