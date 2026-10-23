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

The suite comprises 55 tests across seven modules: the tape's accumulation
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

All Monte Carlo checks use a tolerance of five standard errors of the
estimator under test.

## Benchmarking

```bash
python benchmark.py
```

This script reports two figures: per-call latency (a single `mc_pricer`
invocation covering all simulated paths) and per-path latency (the
per-call figure divided across those paths). Because the two figures
answer different performance questions, both are reported.
