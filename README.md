# AdjointCore

AdjointCore is a reverse-mode Adjoint Algorithmic Differentiation (AAD)
engine, implemented from first principles in Rust. It prices European calls,
European puts, arithmetic-average Asian calls, and smoothed digital and
up-and-out barrier calls by Monte Carlo simulation and, within the same
computation, recovers their Greeks (Delta, Rho, and
one Vega per bucket of a piecewise-constant volatility curve) from a single
backward sweep per simulated path. Paths run in parallel across threads
with results that do not depend on the thread count. The engine is exposed
to Python through PyO3 and `maturin` (currently the European call),
allowing it to be imported and called like any standard Python package once
built.

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
  `{ value: f64, index: u32, generation: u32 }`, which holds no derivative
  data of its own — every partial resides on the tape. The `Gradient`
  struct reads directly from the tape's adjoint buffer rather than
  returning a copy, ensuring `.backward()` performs no allocations.

- **Handle validity**: A `Dual` is valid only on the tape recording it came
  from (same thread, no `tape::reset()` since), and a `Gradient` only until
  the next `backward()` or reset on that thread. Each recording gets a
  process-wide unique generation id, and any use outside that window
  panics with a "stale" message instead of silently reading whatever node
  now occupies the same index. Previously this window was documented for
  `Gradient` but not enforced; the pricer and the existing tests already
  stayed inside it, so their results are unchanged. The cost is one relaxed atomic add per reset and an
  integer compare per recorded operation, sweep, and `wrt` call. The
  generation counter is a `u32` so `Dual` stays 16 bytes; it wraps after
  2^32 resets, so a handle reused exactly that many generations later
  would not be caught.

- **Operators (`src/ops.rs`)**: `Add`, `Sub`, `Mul`, and `Div` are
  implemented across all three relevant operand shapes (`Dual op Dual`,
  `Dual op f64`, `f64 op Dual`), plus unary `Neg`. The engine also supports `exp`, `sqrt`,
  `ln`, `norm_cdf`, `max_zero` (the option payoff kink, whose
  subgradient is taken as 0 exactly at the kink itself), and `step`
  (the indicator 1{x > 0}, recorded with a zero partial, which is all a
  pathwise sweep can see of a jump).

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

- **Payoffs (`src/payoff.rs`)**: A `Payoff` trait gives the number of
  equally spaced fixing dates and the undiscounted payoff on the spot at
  those dates. It is implemented by `EuropeanCall`, `EuropeanPut`,
  `AsianCall` (arithmetic average over the fixings, today's spot excluded),
  and the smoothed `DigitalCall` and `UpAndOutCall` (see
  [Smoothed digital and barrier](#smoothed-digital-and-barrier)).
  Payoffs are written once against a small `Real` scalar trait implemented
  for both `Dual` and `f64`. On `Dual` every step is recorded for the
  adjoint sweep. On `f64` nothing is recorded, which gives a tape-free
  revaluation (`monte_carlo_value`) that sees exactly the same paths for the
  same seed. That revaluation is the bump-and-reval oracle.

- **Monte Carlo pricer (`src/pricer.rs`)**: An Euler-Maruyama simulation of
  geometric Brownian motion, applied to ln S, so each step between fixing
  dates is the exact log-normal transition with no discretization bias.
  `monte_carlo_greeks` is generic over the payoff. Spot, Rate, and
  Volatility are recorded as tape variables, allowing Delta, Rho, and Vega
  to be recovered from the same sweep that produces the price. Time and the
  payoff's own terms (strike, fixings) remain plain constants, since no
  Greek is required with respect to them. The
  standard error of each output quantity is tracked with a streaming
  accumulator (Welford's algorithm). The test suite verifies agreement
  with closed-form Black-Scholes to within five standard errors per
  quantity — a statistically grounded tolerance, rather than one fixed
  bound applied uniformly to all four estimators of differing variance.

- **Volatility curve (`src/pricer.rs`)**: `monte_carlo_greeks_bucketed`
  takes a piecewise-constant volatility curve, with N equal buckets over
  [0, T]. Every bucket volatility is its own tape variable, so one backward
  sweep per path returns the price, Delta, Rho, and all N bucket vegas. The
  simulation grid merges the payoff's fixing dates with the bucket edges
  using exact integer arithmetic (both are integer multiples of T / (nN)),
  so every step lies inside one bucket and stays an exact log-normal
  transition. With one bucket the grid and arithmetic are those of the flat
  pricer, and the results are bit-identical. The flat `monte_carlo_greeks`
  is now that one-bucket case.

- **Parallel paths (`src/pricer.rs`)**: `monte_carlo_greeks_parallel`
  splits the paths into fixed-size chunks and prices them on the current
  rayon thread pool. The chunks are described by `PathChunks { seed,
  chunk_size }`. Each chunk draws from its own `StdRng` (ChaCha12), keyed by
  the seed and the chunk index, so no two chunks share a stream. Each chunk
  records on the thread-local tape of whichever worker runs it, so no tape
  is ever shared. Each chunk keeps its own Welford accumulators. These are
  merged in chunk order with the pairwise formula of Chan, Golub, and
  LeVeque, which combines means weighted by count and corrects the squared
  deviations for the gap between the means. Which paths exist, and the order
  their statistics combine in, depend only on the seed, the chunk size, and
  the path count, so the result is bit-identical on any number of threads.

- **Variance reduction (`src/variance.rs`)**: `monte_carlo_greeks_reduced`
  takes a `VarianceReduction { antithetic, control_variate }`. Both
  reductions apply to the price and to every Greek.
  - **Antithetic.** Each path is paired with its mirror, every normal
    negated. Both legs are recorded on one tape and their average is swept
    once, so a pair costs one backward sweep.
  - **Control variate.** The geometric-average Asian call on the same path
    (Kemna and Vorst), with coefficient 1. Its expectation has a closed form
    even on the piecewise-constant volatility curve, because ln G is normal
    with a mean and variance that accumulate along the same merged grid.
  - **The closed form is on the tape.** `geometric_asian_call` writes it in
    `Dual` arithmetic and records it once, so one sweep gives its exact
    Greeks. The controlled estimator samples payoff minus control along the
    paths and adds those exact Greeks back.
  - **Why a fixed coefficient.** It keeps the estimator unbiased for the
    price and every Greek. An estimated optimal coefficient would add an
    O(1/n) bias.

- **Bump-and-reval (`src/bump.rs`)**: The baseline the adjoint pricer is
  measured against, with no adjoint code involved. It takes central
  differences of the tape-free pricer, re-seeding every revaluation so all
  of them see the same paths, which costs 2N + 5 valuations for N vol
  buckets.

- **PyO3 bridge (`src/python.rs`)**: `mc_pricer` returns
  `{"price", "delta", "vega", "rho"}` for a European Call under a flat
  volatility. `mc_pricer_buckets` takes a list of bucket volatilities and
  returns `{"price", "delta", "rho", "vegas"}`, with `vegas` holding one
  sensitivity per bucket. Once built, the engine is accessible via
  `import adjoint_core`.

## Where the pathwise estimator stops working

Every Greek here is a pathwise estimator: the derivative of each simulated
path's discounted payoff, averaged over paths. It is unbiased only when the
payoff is continuous in the path (Lipschitz is enough), so that
differentiating inside the expectation is allowed. Calls, puts, and
arithmetic Asians qualify. Their kink at the strike has zero probability
under a continuous distribution and contributes nothing.

Discontinuous payoffs such as digitals and barriers do not qualify. Their
pathwise derivative is zero on almost every path, because a small bump does
not change which side of the strike or barrier a path lands on. Yet the true
sensitivity comes entirely from the jump. The pathwise estimate would
therefore be a confident, wrong zero (or a value missing the jump's
contribution). The engine offers these payoffs only with the jump smoothed
into a call spread of width h, which makes the pathwise estimator unbiased
for the smoothed payoff at the cost of an O(h^2) bias against the true one;
[Smoothed digital and barrier](#smoothed-digital-and-barrier) measures that
bias. The likelihood-ratio method and a conditional expectation step are
the unbiased alternatives; this engine does not offer them.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request, on
`ubuntu-latest`. It runs two jobs:

- `cargo test --release`;
- a maturin build check: build the wheel, install it, and import it.

There is no deploy step.

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
cargo test --release
```

The suite comprises 83 tests: one unit test of the accumulator merge and 82
integration tests across twelve modules. They cover the tape's accumulation
and reset behavior, the normal CDF kernel verified against independently
generated reference values, every operator and math function verified
against its analytical derivative, the Monte Carlo engine verified
against closed-form Black-Scholes with an additional cross-seed consistency
check, handle validity (stale `Dual` and `Gradient` use after a reset,
a second sweep, or on another thread is rejected), and the payoffs:
the put against closed-form Black-Scholes and put-call parity on common
paths, and each Asian Greek against central-difference bump-and-reval with
common random numbers. The volatility curve is checked as follows:
- One bucket reproduces the flat pricer bit for bit.
- With equal bucket volatilities, the bucket vegas sum to the flat Vega
  exactly on common paths, and to the Black-Scholes Vega statistically.
- On an unequal curve, the price, Delta, Rho, and each bucket vega match
  Black-Scholes at the integrated-variance volatility.
- Each bucket vega matches bump-and-reval on common random numbers.
- An Asian whose fixings fall between bucket edges agrees with the flat
  Asian.

`bump_and_reval` is checked against the adjoint Greeks on the same seed,
for a European and an Asian call on vol curves, and its price must equal the
adjoint price exactly.

The parallel pricer must give bit-identical results on 1, 2, 4, and 8
threads (20,500 paths, so the last chunk is partial). It must also match
Black-Scholes at the integrated-variance volatility, and two chunk sizes must
give statistically consistent estimates with matching standard errors. The
merged accumulator must equal one sequential pass, including when either
side is empty.

The smoothed payoffs are checked as follows:
- The digital closed form equals minus the strike derivative of the
  Black-Scholes call, for the price and each Greek.
- The smoothed digital's price and Greeks match the closed-form call spread
  (C(K - h/2) - C(K + h/2)) / h at three strikes and two widths, so the
  pathwise estimator is unbiased for the payoff it smooths.
- The smoothing bias of the digital falls by a factor of 4 when h halves.
- On the exact digital, the pathwise Delta and Vega have mean 0 and standard
  error 0 (zero on every path), and Rho is -T times the price.
- An unreachable exact barrier reproduces the European call bit for bit.
- The smoothed barrier's Delta, Rho, and bucket vegas match bump-and-reval
  of the same smoothed payoff on common random numbers.

The netting-set CVA (`src/xva.rs`) is checked as follows:
- Netting never raises CVA or expected exposure against the sum of the
  trades' standalone values on common paths, and a fully offset set has CVA
  and every sensitivity exactly zero.
- A long-only set matches its closed form: its value is never negative, so
  the expected discounted exposure at each date is today's value of the
  trades still alive. CVA, Delta, Rho, each bucket vega, the hazard-rate
  sensitivity, and the expected exposure profile are all checked.
- With equal bucket vols, the bucket vegas sum to the flat Vega on common
  paths.
- A set's sensitivity to another counterparty's hazard rate is exactly zero,
  and the total across sets is the sum of the sets.
- The tape-free CVA reproduces the adjoint CVA bit for bit.
- On mixed long and short sets, every sensitivity matches bump-and-reval on
  common random numbers.

All Monte Carlo checks use a tolerance of five standard errors of the
estimator under test.

## Benchmarking

### AAD against bump-and-reval as the sensitivity count grows

```bash
cargo run --release --example scaling
```

`examples/scaling.rs` prices an arithmetic Asian call (100 fixings, T = 1,
S = K = 100, r = 5%, flat 20% vol split into N buckets) on 10,000 paths with
seed 42. It returns N + 2 sensitivities (Delta, Rho, N bucket vegas) two
ways:

- the adjoint pricer, with one backward sweep per path;
- `bump_and_reval` (`src/bump.rs`), with central differences of the
  tape-free pricer. Every revaluation is re-seeded so all of them see the
  same paths, which gives 2N + 5 valuations in all.

Both methods use the same paths, seed, and 100-step grid. Every N divides
100, so one valuation costs the same at every N. Each N gets one discarded
warm-up round, then seven timed rounds, each running the three methods back
to back. The table shows the median and [min, max] wall time.

Measured on 2026-10-23: Apple M4, 10 cores, 16 GB, macOS (Darwin 24.1.0),
rustc 1.97.1, single-threaded, release profile.

| N | sensitivities | valuation ms | AAD ms | bump ms | AAD / bump | AAD / valuation | bump / valuation | tape nodes / path | max gap (SE) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 13.8 | 39.2 [39.1, 39.3] | 97.4 [96.9, 102.0] | 0.403 | 2.83 | 7.0 | 512 | 0.027 |
| 2 | 4 | 13.9 | 39.7 [39.6, 42.1] | 128.2 [125.5, 144.1] | 0.310 | 2.85 | 9.2 | 518 | 0.027 |
| 5 | 7 | 14.1 | 40.9 [40.8, 41.3] | 212.4 [211.0, 213.5] | 0.193 | 2.91 | 15.1 | 536 | 0.027 |
| 10 | 12 | 14.5 | 43.0 [42.8, 45.3] | 361.4 [358.4, 372.7] | 0.119 | 2.97 | 25.0 | 566 | 0.027 |
| 25 | 27 | 14.3 | 48.1 [47.9, 49.5] | 801.4 [793.2, 880.1] | 0.060 | 3.36 | 55.9 | 656 | 0.027 |
| 50 | 52 | 15.0 | 56.0 [55.9, 57.4] | 1563.2 [1556.2, 1583.6] | 0.036 | 3.75 | 104.5 | 806 | 0.027 |
| 100 | 102 | 15.2 | 73.6 [73.6, 75.9] | 3151.1 [3142.9, 3193.1] | 0.023 | 4.84 | 207.1 | 1106 | 0.027 |

What the numbers say:

- **Bump-and-reval scales with N, as theory predicts.** Its cost is
  2N + 5 valuations. The measured ratio to one valuation is 7.0 at N = 1 and
  207.1 at N = 100, against 7 and 205.
- **AAD does not scale with N, but it is not perfectly flat either.** Going
  from 3 to 102 sensitivities raised its cost from 2.83 to 4.84 valuations
  (39.2 to 73.6 ms). The reason is the tape. Each bucket adds one input node
  plus five nodes for its drift and diffusion terms, so the tape grows from
  512 to 1106 nodes per path, exactly 6 per extra bucket. Each extra vega
  also costs one adjoint lookup and one accumulator update per path. Time
  grows smoothly with tape length, with no step that would point to a cache
  effect. At 1106 nodes the tape and its adjoints are about 35 KB per path
  (24-byte nodes plus 8-byte adjoints), a figure derived from the struct
  sizes rather than measured.
- **At N = 100, AAD returns 102 sensitivities 43x faster than
  bump-and-reval** (73.6 ms against 3151.1 ms).
- **Both methods compute the same Greeks.** The largest gap between any
  adjoint Greek and its bump-and-reval value is 0.027 of the adjoint
  estimator's standard error at every N.
- **Run-to-run variation.** A second run of the same command gave an AAD
  median of 73.5 ms and a bump median of 3214.1 ms at N = 100. An earlier
  run, made before the tape column was added and with other load on the
  machine, gave a bump median of 3625.1 ms with a max of 5555.7 ms. The AAD
  column was stable within about 1 ms across all three runs.

### Parallel paths

```bash
cargo run --release --example parallel
```

`examples/parallel.rs` prices the same Asian call on a 100-bucket curve (102
sensitivities) on 100,000 paths in chunks of 1,000, with seed 42. Each
thread count runs on a dedicated rayon pool of that size, with one discarded
warm-up and five timed runs. Measured on 2026-10-23 on the same Apple M4,
which has 4 performance and 6 efficiency cores:

| threads | AAD ms (median [min, max]) | speedup vs 1 thread | identical to 1 thread |
|---|---|---|---|
| 1 | 757.5 [755.5, 758.5] | 1.00 | true |
| 2 | 386.6 [384.7, 391.5] | 1.96 | true |
| 4 | 207.2 [196.4, 226.5] | 3.66 | true |
| 8 | 119.2 [118.3, 138.5] | 6.35 | true |
| 10 | 107.1 [100.3, 131.4] | 7.07 | true |

The serial pricer, with no chunks and no pool, took 758.2 ms
[757.9, 758.4], so chunking costs nothing measurable. Scaling is close to
linear up to 4 threads. Beyond that, the extra threads land on efficiency
cores, which are slower, so 10 threads give 7.07x rather than 10x. A second
run gave 1.97x, 3.73x, 6.35x, and 7.09x at 2, 4, 8, and 10 threads. Every
parallel result was bit-identical to the one-thread result.

### Variance reduction

```bash
cargo run --release --example variance_reduction
```

`examples/variance_reduction.rs` prices a 12-fixing arithmetic Asian call
(S = 100, r = 5%, T = 1, 5-bucket vol curve 18/22/20/25/15%) at three
strikes. Every method uses 100,000 simulated paths from seed 42 (antithetic:
50,000 pairs). VR is the variance reduction factor, (plain SE / method SE)^2:
how many times more plain paths the same standard error would need. The
vegas column gives the smallest and largest VR across the five buckets. Time
is the median of five runs. Efficiency is the price VR times
(plain time / method time). Measured on 2026-10-24 on the same Apple M4.

| K | method | price | price SE | VR price | VR delta | VR rho | VR vegas [min, max] | ms | efficiency |
|---|---|---|---|---|---|---|---|---|---|
| 90 | plain | 12.9684 | 0.03484 | 1.0 | 1.0 | 1.0 | [1.0, 1.0] | 170.2 | 1.0 |
| 90 | antithetic | 12.9206 | 0.01282 | 7.4 | 2.2 | 0.9 | [2.5, 3.3] | 147.7 | 8.5 |
| 90 | control | 12.9202 | 0.00123 | 808.1 | 29.9 | 14.6 | [52.0, 203.4] | 181.7 | 756.9 |
| 90 | both | 12.9183 | 0.00116 | 909.9 | 29.4 | 15.1 | [47.1, 214.4] | 182.0 | 851.2 |
| 100 | plain | 6.1918 | 0.02708 | 1.0 | 1.0 | 1.0 | [1.0, 1.0] | 140.3 | 1.0 |
| 100 | antithetic | 6.1654 | 0.01879 | 2.1 | 6.9 | 3.6 | [1.2, 1.3] | 131.5 | 2.2 |
| 100 | control | 6.1644 | 0.00127 | 455.1 | 40.7 | 26.4 | [32.6, 84.2] | 194.9 | 327.6 |
| 100 | both | 6.1628 | 0.00106 | 658.2 | 39.9 | 27.3 | [34.0, 82.3] | 151.1 | 611.3 |
| 110 | plain | 2.3241 | 0.01723 | 1.0 | 1.0 | 1.0 | [1.0, 1.0] | 104.5 | 1.0 |
| 110 | antithetic | 2.3056 | 0.01559 | 1.2 | 1.6 | 1.6 | [1.0, 1.1] | 99.4 | 1.3 |
| 110 | control | 2.3057 | 0.00126 | 186.2 | 28.5 | 20.9 | [18.4, 56.8] | 124.3 | 156.5 |
| 110 | both | 2.3042 | 0.00114 | 228.3 | 28.3 | 21.3 | [18.6, 60.8] | 126.8 | 188.3 |

What the numbers say:

- **The control variate does almost all the work.** It cuts the price
  variance 186x to 808x, Delta and Rho variance 15x to 41x, and bucket-vega
  variance 18x to 203x. Its extra cost is 7% to 39% more time, from one
  logarithm per fixing on the tape plus the subtraction, so per unit of
  compute it is worth 157x to 757x on the price.
- **The reduction depends on the strike.** It is largest in the money,
  where both averages are almost always above the strike and their payoffs
  move together nearly linearly. It is smallest out of the money, where the
  kink sits in a different place for the arithmetic and the geometric
  average.
- **Antithetic sampling is weak on its own, and can hurt.** It reduces price
  variance 1.2x to 7.4x, and less out of the money, where both legs of most
  pairs pay zero. On the K = 90 Rho it is slightly worse than plain
  (VR 0.9): mirroring only helps a quantity that moves roughly monotonically
  with the normals, which that pathwise estimator does not.
- **Combining the two mostly helps the price.** The combination adds up to
  1.45x on the price over the control alone, and leaves the Greeks about
  where the control put them.
- **Correctness.** All four methods' Greeks agree with each other within
  their combined standard errors (`tests/variance_reduction.rs`). With one
  fixing, the control is the European call itself, so the controlled
  estimator returns Black-Scholes with a standard error below 1e-10.
- **Repeatability.** A second run gave identical VR columns (fixed seed).
  Times differed by up to 32 ms per method; the largest gap was K = 100
  control, 194.9 ms against 162.9 ms.

### Smoothed digital and barrier

```bash
cargo run --release --example smoothing
```

`DigitalCall { strike, smoothing }` pays 1 if S_T > K. `UpAndOutCall {
strike, barrier, fixings, smoothing }` is a call that is knocked out if the
spot is at or above the barrier on any of its equally spaced fixing dates.
Both replace each indicator with a centered call spread of width h:
1{x > 0} becomes ((x + h/2)+ - (x - h/2)+) / h, a ramp from 0 to 1 across
the band [-h/2, h/2]. The smoothed payoff is Lipschitz in the path, so the
pathwise estimator is unbiased for it, and biased against the true payoff by
the smoothing alone. With h = 0 the payoff is exact: its price is unbiased
and its pathwise Greeks are not.

- **Bias is O(h^2).** The ramp is the indicator averaged over shifts in
  [-h/2, h/2]. That average is symmetric, so the first-order term cancels. A
  one-sided spread, the conservative overhedge, would give O(h).
- **Variance is O(1/h).** Inside the band the payoff's slope is 1/h, and a
  path lands in the band with probability proportional to h. The variance of
  a pathwise Greek therefore grows like 1/h, and its SE like 1/sqrt(h).

Setup: S = K = 100, r = 5%, flat 20% vol, T = 1, seed 42. A bias is the
Monte Carlo estimate minus the reference, and z is the bias divided by its
standard error. Every row of a table reuses the same paths, so the noise in
a column is correlated down the rows: read a column for its trend. Measured
on 2026-10-24 on the same Apple M4; the whole example takes 57 s.

**Digital call, 1,000,000 paths.** The reference is the Black-Scholes
cash-or-nothing closed form (`black_scholes_digital`), which is exact: price
0.53232, Delta 0.01876, Vega -0.6567, Rho 1.3439. The smoothed payoff's own
expectation is (C(K - h/2) - C(K + h/2)) / h, also closed form. Delta exact
bias is therefore the smoothing bias with no noise in it, and Delta RMSE is
sqrt(exact bias^2 + SE^2) at this path count. The last column is the
largest |MC - call spread| / SE over the price and the three Greeks.

| h | price bias (z) | Delta | Delta bias (z) | Delta SE | Delta exact bias | Delta RMSE | Vega bias (z) | Vega SE | Rho bias (z) | gap to spread (max abs z) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.00038 (-0.8) | 0.00000 | -0.01876 (-) | 0.00000 | -1.9e-2 | 0.01876 | +0.6567 (-) | 0.0000 | -1.8758 (-3972.0) | - |
| 0.5 | -0.00041 (-0.9) | 0.01868 | -0.00008 (-0.4) | 0.00019 | -4.9e-7 | 0.00019 | +0.0030 (+0.5) | 0.0066 | -0.0079 (-0.4) | 0.87 |
| 1 | -0.00042 (-0.9) | 0.01875 | -0.00002 (-0.1) | 0.00013 | -2.0e-6 | 0.00013 | +0.0007 (+0.2) | 0.0046 | -0.0012 (-0.1) | 0.90 |
| 2 | -0.00042 (-0.9) | 0.01881 | +0.00005 (+0.5) | 0.00009 | -7.9e-6 | 0.00009 | -0.0020 (-0.6) | 0.0033 | +0.0054 (+0.6) | 0.91 |
| 5 | -0.00034 (-0.7) | 0.01872 | -0.00004 (-0.7) | 0.00006 | -4.9e-5 | 0.00007 | +0.0019 (+0.9) | 0.0020 | -0.0036 (-0.6) | 0.85 |
| 10 | -0.00017 (-0.4) | 0.01859 | -0.00017 (-4.6) | 0.00004 | -2.0e-4 | 0.00020 | +0.0076 (+5.3) | 0.0014 | -0.0172 (-4.5) | 0.89 |
| 20 | +0.00042 (+1.0) | 0.01801 | -0.00075 (-32.4) | 0.00002 | -7.6e-4 | 0.00076 | +0.0345 (+31.1) | 0.0011 | -0.0754 (-31.6) | 1.35 |

On the exact digital (h = 0), over 1,000,000 paths, the pathwise Delta and
Vega have mean 0 and standard error 0, which means they are 0 on every path.
The pathwise Rho is -T times the price on every path, because only the
discount factor depends on the rate: -0.532 against a true 1.344.

**Up-and-out call, barrier 120, 12 monthly fixings, 1,000,000 paths** (the
parallel pricer, chunks of 10,000). Neither reference has a closed form:

- **Price reference.** The unsmoothed Monte Carlo price, 1.84857 (SE
  0.00092), on 20,000,000 paths with seed 7. The price smoothing bias is
  measured on those same paths, as the mean of the per-path difference
  between the smoothed and the exact payoff, so its SE is that of the
  difference alone.
- **Greek reference.** Central-difference bump-and-reval of the unsmoothed
  payoff on common random numbers (`bump_and_reval`), averaged over 128
  independent seeds of 1,000,000 paths, with the SE taken across seeds:
  Delta -0.00827 (SE 0.00131), Vega -14.890 (SE 0.100), Rho 1.989
  (SE 0.118). Each revaluation sees the jump, so it is unbiased for the true
  Greeks up to O(bump^2). The price of that is a variance of order 1/bump,
  which is why it takes 128 million paths. The Greek z combines the
  estimator's SE with the reference's; they are independent (different
  seeds).

| h | price smoothing bias (z) | Delta | Delta bias (z) | Delta SE | Vega | Vega bias (z) | Vega SE | Rho | Rho bias (z) | Rho SE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 (exact) | 0.2473 | +0.2555 (+184.7) | 0.0004 | 1.094 | +15.984 (+158.2) | 0.013 | 22.878 | +20.889 (+168.0) | 0.041 |
| 0.5 | -0.00044 (-5.2) | -0.0097 | -0.0014 (-0.4) | 0.0031 | -14.967 | -0.077 (-0.4) | 0.190 | 1.868 | -0.121 (-0.4) | 0.274 |
| 1 | -0.00171 (-14.5) | -0.0117 | -0.0035 (-1.4) | 0.0022 | -15.063 | -0.173 (-1.0) | 0.134 | 1.658 | -0.331 (-1.5) | 0.196 |
| 2 | -0.00716 (-43.1) | -0.0109 | -0.0027 (-1.3) | 0.0016 | -14.950 | -0.060 (-0.4) | 0.093 | 1.771 | -0.219 (-1.2) | 0.139 |
| 5 | -0.04398 (-167.5) | -0.0128 | -0.0045 (-2.7) | 0.0010 | -14.547 | +0.343 (+3.0) | 0.056 | 1.641 | -0.348 (-2.4) | 0.089 |
| 10 | -0.16565 (-441.2) | -0.0153 | -0.0070 (-4.7) | 0.0007 | -13.140 | +1.751 (+16.5) | 0.035 | 1.556 | -0.433 (-3.3) | 0.063 |

What the numbers say:

- **Unsmoothed pathwise Greeks are wrong, not noisy.** On the digital, Delta
  and Vega are zero on every path. On the barrier, the pathwise Delta is
  +0.247 against a true -0.008, Vega +1.09 against -14.89, and Rho 22.9
  against 2.0, each more than 150 SE away. The pathwise estimate is the
  call's Greek on the surviving paths. It misses the value lost at the
  knock-out, which dominates.
- **Smoothing bias scales as h^2.** The digital's exact Delta bias is 4.9e-7,
  2.0e-6, 7.9e-6, 4.9e-5, 2.0e-4, and 7.6e-4 for h = 0.5 to 20, a factor of
  about 4 per doubling (6.2 from h = 2 to 5, against 6.25). The barrier
  price bias is -0.00044, -0.00171, -0.00716, -0.0440, and -0.1657 for
  h = 0.5 to 10, factors of 3.9, 4.2, 6.1 (against 6.25), and 3.8.
- **SE scales as 1/sqrt(h).** The barrier Delta SE is 0.0031 at h = 0.5 and
  0.0007 at h = 10, a ratio of 4.4 against sqrt(20) = 4.47.
- **The optimum width depends on the path count.** For the digital at
  1,000,000 paths, Delta RMSE is lowest at h = 5 (0.00007: SE 0.00006, bias
  -4.9e-5). It is 0.00019 at h = 0.5, where noise dominates, and 0.00076 at
  h = 20, where bias dominates. More paths shrink the SE and move the optimum
  to a smaller h.
- **The estimator is unbiased for what it smooths.** Every smoothed digital
  row sits within 1.35 SE of the call-spread closed form on all four
  outputs, so the measured bias is the smoothing bias. For the barrier, the
  Greek bias is below the reference's resolution up to h = 2 (|z| at most
  1.5). It is resolved at h = 5 and 10: Vega +0.34 (z 3.0) and +1.75
  (z 16.5), and Delta -0.0070 (z -4.7) at h = 10.
- **The barrier Delta is close to zero.** Here the gain from a higher spot
  and the loss from a likelier knock-out nearly cancel (true Delta
  -0.008 +- 0.0013). Relative errors in Delta are large, and Vega is the
  clearer Greek to read the bias from.
- **The rows share their price noise.** The digital price bias of about
  -0.0004 (z -0.9) in every row is the noise of those 1,000,000 paths, not
  bias.
- **Repeatability.** A second run gave identical output (fixed seeds,
  parallel results merged in chunk and seed order).

### Python call latency

```bash
python benchmark.py
```

This script reports two figures: per-call latency (a single `mc_pricer`
invocation covering all simulated paths) and per-path latency (the
per-call figure divided across those paths). Because the two figures
answer different performance questions, both are reported.
