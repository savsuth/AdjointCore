# AdjointCore

AdjointCore is a reverse-mode Adjoint Algorithmic Differentiation (AAD)
engine, implemented from first principles in Rust. It prices European calls,
European puts, and arithmetic-average Asian calls by Monte Carlo simulation
and, within the same computation, recovers their Greeks (Delta, Vega, and
Rho) from a single backward sweep per simulated path. The engine is exposed
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

- **Payoffs (`src/payoff.rs`)**: A `Payoff` trait gives the number of
  equally spaced fixing dates and the undiscounted payoff on the spot at
  those dates. It is implemented by `EuropeanCall`, `EuropeanPut`, and
  `AsianCall` (arithmetic average over the fixings, today's spot excluded).
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
contribution), and the engine deliberately does not offer such payoffs. They
need a smoothed payoff, the likelihood-ratio method, or a conditional
expectation step instead.

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

The suite comprises 58 tests across eight modules: the tape's accumulation
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

### Python call latency

```bash
python benchmark.py
```

This script reports two figures: per-call latency (a single `mc_pricer`
invocation covering all simulated paths) and per-path latency (the
per-call figure divided across those paths). Because the two figures
answer different performance questions, both are reported.
