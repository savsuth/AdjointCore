"""Latency benchmark for the compiled `adjoint_core` extension.

"Latency per pricing call" could mean two different things, so we report
both: per-call is one `mc_pricer(...)` invocation covering every `num_paths`,
and per-path is that divided by `num_paths` — basically the cost of a single
tape record, backward sweep, and reset, which is the number that actually
speaks to the zero-allocation arena design.
"""

import statistics
import time

import adjoint_core

SPOT, STRIKE, RATE, VOL, TIME = 42.0, 40.0, 0.1, 0.2, 0.5
NUM_PATHS = 100_000
TRIALS = 11
WARMUP = 2


def run_once() -> float:
    start = time.perf_counter()
    adjoint_core.mc_pricer(
        spot=SPOT,
        strike=STRIKE,
        rate=RATE,
        vol=VOL,
        time=TIME,
        num_paths=NUM_PATHS,
    )
    return time.perf_counter() - start


if __name__ == "__main__":
    for _ in range(WARMUP):
        run_once()

    samples_s = [run_once() for _ in range(TRIALS)]
    samples_ms = sorted(s * 1_000 for s in samples_s)

    median_ms = statistics.median(samples_ms)
    mean_ms = statistics.mean(samples_ms)
    stdev_ms = statistics.stdev(samples_ms)
    min_ms = samples_ms[0]
    max_ms = samples_ms[-1]

    per_path_ns = (median_ms / NUM_PATHS) * 1_000_000

    print(f"paths per call:      {NUM_PATHS:,}")
    print(f"trials:              {TRIALS} ({WARMUP} warmup, discarded)")
    print()
    print(f"per-call latency:    median {median_ms:.3f} ms   "
          f"mean {mean_ms:.3f} ms   stdev {stdev_ms:.3f} ms   "
          f"[{min_ms:.3f}, {max_ms:.3f}] ms")
    print(f"per-path latency:    {per_path_ns:.1f} ns/path  "
          f"(median per-call / {NUM_PATHS:,})")
