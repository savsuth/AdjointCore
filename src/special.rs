//! Double-precision scalar kernels for the standard normal distribution.
//!
//! These are plain `f64` routines — no tape involved. `Dual::norm_cdf` in
//! [`crate::ops`] is the one that calls them and records the derivative.
//!
//! # Why we didn't just use Hart/West
//!
//! The rational approximation most quant libraries reach for gets called
//! "double precision," but its actual accuracy is a bit more modest than
//! that once you check it against 60-digit references: the rational branch
//! is off by 2.6e-9 near |x| = 7, and its tail is off by 8.7e-9 near
//! |x| = 8. That's roughly 1e-9 grade, which sits right on top of our 1e-8
//! Black-Scholes tolerance — a passing Greeks test wouldn't actually mean
//! anything.
//!
//! So instead we work directly in `x` (never through `erfc(x/sqrt(2))`,
//! which costs a factor of `2x` in tail accuracy just from the argument
//! scaling), and we split the range at |x| = 1.5: an all-positive-term `erf`
//! series handles everything below that, and the Mills-ratio continued
//! fraction (modified Lentz) handles everything above it. The key difference
//! from Hart/West is that we run the fraction to convergence instead of
//! truncating it at a fixed depth — that fixed-depth truncation is exactly
//! where their tail error comes from.
//!
//! Checked against mpmath at 60 digits, swept every 0.005: worst case is
//! 2.6e-15 (lower tail, around x = -2.0), and 1.7e-16 everywhere else. That
//! 2.6e-15 isn't the approximation being imprecise — it's just rounding
//! piling up over roughly 150 Lentz iterations near the split. Outside
//! |x| in [1.3, 2.5], including the deep tail, we're sitting at the 1e-16
//! floor.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

/// 1 / sqrt(2*pi)
const INV_SQRT_2PI: f64 = 0.3989422804014326779399460599344;

/// sqrt(2*pi), the normalising constant of the Mills-ratio tail.
const SQRT_2PI: f64 = 2.5066282746310005024157652848110;

/// Where we switch from the series to the continued fraction.
///
/// Picked by just measuring both sides: below this, the `1 - erf`
/// cancellation stays mild enough to keep us near the precision floor; above
/// it, the continued fraction converges in a reasonable number of steps.
const SERIES_SPLIT: f64 = 1.5;

/// Past this |x|, the true result is subnormal and there's no relative
/// precision left to preserve, so we just flush to 0 or 1. N(-37.5) is about
/// 4.6e-308 — still technically a normal float, but only just.
const TAIL_CUTOFF: f64 = 37.5;

/// Guard against a zero denominator in the Lentz recurrence.
const TINY: f64 = 1e-300;

const MAX_SERIES_TERMS: u32 = 128;
const MAX_CF_TERMS: u32 = 512;

/// exp(-y^2/2), computed with the exponent split so rounding can't sneak in.
///
/// If you just write `(-0.5 * y * y).exp()`, you lose precision way out in
/// the tail: rounding `y * y` introduces an absolute error of about
/// `ulp(y^2)`, and `exp` turns that straight into relative error — by
/// y = 34 that's already 6e-14, which is bad.
///
/// The fix: truncate `y` down to the nearest 1/16, so `y_hi^2` is exactly
/// representable with no rounding at all, and let only the small leftover
/// remainder carry whatever rounding happens. Same trick Cody uses in
/// CALERF.
fn gauss_exp(y: f64) -> f64 {
    let y_hi = (y * 16.0).trunc() / 16.0;
    let remainder = (y - y_hi) * (y + y_hi);
    (-0.5 * y_hi * y_hi).exp() * (-0.5 * remainder).exp()
}

/// erf(t) for small non-negative `t`, using the confluent hypergeometric form
///
/// ```text
/// erf(t) = (2/sqrt(pi)) * exp(-t^2) * sum_k (2 t^2)^k / (1*3*5*...*(2k+1))
/// ```
///
/// We pick this form over the usual alternating Maclaurin series specifically
/// because every term here is positive — nothing cancels on the way to the
/// sum. We still compensate the summation (Kahan-style) because the caller
/// turns right around and forms `1 - erf`, which will amplify whatever error
/// we leave behind.
fn erf_series(t: f64) -> f64 {
    let t_squared = t * t;

    let mut term = t;
    let mut sum = t;
    let mut compensation = 0.0f64;
    let mut k: u32 = 0;

    while term.abs() > f64::EPSILON * sum.abs() && k < MAX_SERIES_TERMS {
        k += 1;
        term *= 2.0 * t_squared / (2.0 * k as f64 + 1.0);

        let adjusted = term - compensation;
        let next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
    }

    FRAC_2_SQRT_PI * (-t_squared).exp() * sum
}

/// The Mills-ratio continued fraction
///
/// ```text
/// R(y) = y + 1/(y + 2/(y + 3/(y + 4/(y + ...))))
/// ```
///
/// We evaluate this with modified Lentz, which walks forward and checks its
/// own convergence instead of just stopping after a fixed number of terms.
/// That distinction actually matters a lot here: truncating at the classic
/// four levels only gets you to 9e-9, but letting it run to convergence gets
/// you all the way down to the floating-point floor. And it's not even slow —
/// it costs about 180 iterations right at the split, but that drops to 17 by
/// y = 7 and just 8 by y = 20.
fn mills_ratio(y: f64) -> f64 {
    let mut f = y;
    let mut c = f;
    let mut d = 0.0f64;

    for j in 1..MAX_CF_TERMS {
        let a = j as f64;

        d = y + a * d;
        if d == 0.0 {
            d = TINY;
        }
        c = y + a / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;

        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }

    f
}

/// Standard normal density, phi(x).
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * gauss_exp(x.abs())
}

/// Standard normal cumulative distribution, N(x).
///
/// We always compute the upper tail directly and only flip it at the very
/// end, so the lower tail keeps its full *relative* accuracy instead of
/// getting formed by subtracting from 1 (which would wreck it). That matters
/// here because deep out-of-the-money options can have N(d2) as small as
/// 1e-16 and it's still driving a real sensitivity.
pub fn norm_cdf(x: f64) -> f64 {
    let abs_x = x.abs();

    if abs_x > TAIL_CUTOFF {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }

    let upper_tail = if abs_x < SERIES_SPLIT {
        0.5 * (1.0 - erf_series(abs_x * FRAC_1_SQRT_2))
    } else {
        gauss_exp(abs_x) / (mills_ratio(abs_x) * SQRT_2PI)
    };

    if x > 0.0 {
        1.0 - upper_tail
    } else {
        upper_tail
    }
}
