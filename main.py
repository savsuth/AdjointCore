"""A quick smoke test for the compiled `adjoint_core` extension.

We price Hull's textbook case here (S=42, K=40, r=0.1, vol=0.2, T=0.5), which
should give an analytical call price of 4.76. The Rust suite already checks
this properly against closed-form and 100,000-path Monte Carlo at 5-sigma
(see tests/phase3_pricer.rs) — this script is just here to confirm the
compiled extension gives back the same shape of answer once it's crossed the
FFI boundary into Python.
"""

import adjoint_core

if __name__ == "__main__":
    result = adjoint_core.mc_pricer(
        spot=42.0,
        strike=40.0,
        rate=0.1,
        vol=0.2,
        time=0.5,
        num_paths=100_000,
    )

    print(result)
    print(f"price = {result['price']:.4f}  (analytical: 4.76)")
    print(f"delta = {result['delta']:.4f}")
    print(f"vega  = {result['vega']:.4f}")
    print(f"rho   = {result['rho']:.4f}")
