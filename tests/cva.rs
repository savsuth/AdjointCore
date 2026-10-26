//! Unilateral CVA of netting sets of European options, with every
//! sensitivity from one recording per path and one backward sweep per
//! netting set.
//!
//! References, and what each one is a reference for:
//! - Netting: max(sum v_i, 0) <= sum max(v_i, 0) on every path, so a set's
//!   CVA and expected exposure never exceed the sum of its trades'
//!   standalone ones on common paths; a fully offset set has none at all.
//! - Long-only set: V >= 0 on every path, so max(V, 0) = V, and D(t) V(t)
//!   is a martingale. The expected discounted exposure at each date is then
//!   today's value of the trades still alive, which makes CVA, every
//!   sensitivity and the expected exposure profile closed form (Black-Scholes
//!   at the integrated-variance vol).
//! - Equal bucket vols on a grid whose bucket edges are exposure dates: the
//!   same paths as the flat vol, so by the chain rule the bucket vegas sum to
//!   the flat Vega on every path.
//! - Bookkeeping: a set's CVA does not depend on another set's hazard rate,
//!   so that sensitivity is exactly zero, and the total is the sum of the
//!   sets path by path.
//! - The tape-free valuation sees exactly the adjoint pricer's paths and
//!   arithmetic, which is what makes it the bump-and-reval oracle.
//! - Mixed long/short sets: each adjoint sensitivity against central-
//!   difference bump-and-reval of the tape-free CVA on common random numbers.
//!   The exposure's kink at V = 0 has probability zero, so the pathwise
//!   estimator is unbiased and the two agree within its standard error.

use adjoint_core::{
    CvaGreeks, CvaResult, Greeks, Market, NettingSet, OptionKind, Trade, black_scholes,
    black_scholes_put, bump_and_reval_cva, cva_value, monte_carlo_cva,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 42;
const SIGMA: f64 = 5.0;
const NUM_PATHS: u64 = 100_000;
const EXPOSURE_DATES: usize = 8;
const VOLS: [f64; 4] = [0.18, 0.22, 0.25, 0.20];
const MARKET: Market = Market {
    spot: 100.0,
    rate: 0.05,
    vols: &VOLS,
    time: 2.0,
};
const HAZARD: f64 = 0.02;
const RECOVERY: f64 = 0.4;

fn assert_within_mc_tolerance(name: &str, mc: f64, reference: f64, std_error: f64) {
    let error = (mc - reference).abs();
    let tolerance = SIGMA * std_error;
    assert!(
        error < tolerance,
        "{name}: MC = {mc}, reference = {reference}, error = {error:e} \
         exceeds {SIGMA} standard errors ({tolerance:e}, std_error = {std_error:e})"
    );
}

fn set(trades: &[Trade]) -> NettingSet {
    NettingSet {
        trades: trades.to_vec(),
        hazard_rate: HAZARD,
        recovery: RECOVERY,
    }
}

fn cva_of(sets: &[NettingSet], market: &Market) -> CvaResult {
    let mut rng = StdRng::seed_from_u64(SEED);
    monte_carlo_cva(sets, market, EXPOSURE_DATES, NUM_PATHS, &mut rng)
}

const LONG_CALL: Trade = Trade {
    kind: OptionKind::Call,
    strike: 100.0,
    maturity: 2.0,
    notional: 1.0,
};
const SHORT_PUT: Trade = Trade {
    kind: OptionKind::Put,
    strike: 95.0,
    maturity: 1.5,
    notional: -1.0,
};
const LONG_PUT: Trade = Trade {
    kind: OptionKind::Put,
    strike: 90.0,
    maturity: 1.0,
    notional: 2.0,
};

/// One run prices the netted set and each trade on its own, so all of them
/// see the same paths and the per-path inequality carries over to the means
/// (up to rounding). The short put is worth less than nothing on its own,
/// so netting it against the long call strictly lowers the exposure. A long
/// and a short of the same call net to exactly zero on every path: no
/// exposure, no CVA, and every sensitivity exactly zero.
#[test]
fn netting_never_raises_cva_and_an_offset_set_has_none() {
    let short_call = Trade {
        notional: -1.0,
        ..LONG_CALL
    };
    let sets = [
        set(&[LONG_CALL, SHORT_PUT, LONG_PUT]),
        set(&[LONG_CALL]),
        set(&[SHORT_PUT]),
        set(&[LONG_PUT]),
        set(&[LONG_CALL, short_call]),
    ];
    let result = cva_of(&sets, &MARKET);
    let s = &result.sets;
    assert_eq!(s.len(), sets.len());

    let netted = s[0].greeks.cva;
    let standalone = s[1].greeks.cva + s[2].greeks.cva + s[3].greeks.cva;
    assert!(netted > 0.0, "netted CVA {netted}");
    assert!(
        netted < standalone,
        "netted CVA {netted} should be below the standalone sum {standalone}"
    );
    assert_eq!(s[2].greeks.cva, 0.0, "a short put alone has no exposure");
    assert_eq!(s[0].expected_exposure.len(), EXPOSURE_DATES);
    for j in 0..EXPOSURE_DATES {
        let ee = s[0].expected_exposure[j];
        let sum = s[1].expected_exposure[j] + s[2].expected_exposure[j] + s[3].expected_exposure[j];
        assert!(
            ee <= sum * (1.0 + 1e-12),
            "date {j}: netted exposure {ee} above standalone sum {sum}"
        );
    }

    let offset = &s[4];
    for (name, g) in [("value", &offset.greeks), ("std error", &offset.std_error)] {
        assert_eq!(g.cva, 0.0, "offset {name} cva");
        assert_eq!(g.delta, 0.0, "offset {name} delta");
        assert_eq!(g.rho, 0.0, "offset {name} rho");
        assert!(g.vegas.iter().all(|&v| v == 0.0), "offset {name} vegas");
        assert!(g.hazards.iter().all(|&h| h == 0.0), "offset {name} hazards");
    }
    assert!(offset.expected_exposure.iter().all(|&e| e == 0.0));
}

/// Today's Black-Scholes value and Greeks of `trade` on the bucket curve:
/// the flat closed form at the integrated-variance vol over [0, maturity],
/// with each bucket vega Vega * vol_k * (overlap of bucket k) / (vol_eff T).
fn closed_form(trade: &Trade, market: &Market) -> (Greeks, Vec<f64>) {
    let width = market.time / market.vols.len() as f64;
    let overlaps: Vec<f64> = (0..market.vols.len())
        .map(|k| (trade.maturity - k as f64 * width).clamp(0.0, width))
        .collect();
    let variance: f64 = market
        .vols
        .iter()
        .zip(&overlaps)
        .map(|(v, l)| v * v * l)
        .sum();
    let vol_eff = (variance / trade.maturity).sqrt();
    let bs = match trade.kind {
        OptionKind::Call => black_scholes,
        OptionKind::Put => black_scholes_put,
    };
    let g = bs(
        market.spot,
        trade.strike,
        market.rate,
        vol_eff,
        trade.maturity,
    );
    let vegas = market
        .vols
        .iter()
        .zip(&overlaps)
        .map(|(v, l)| g.vega * v * l / (vol_eff * trade.maturity))
        .collect();
    (g, vegas)
}

/// A long-only set's CVA, sensitivities and expected exposure profile in
/// closed form. Trade i counts on the dates t_j < T_i, and the default
/// probabilities over those dates telescope to 1 - e^{-lambda t_last}.
fn long_only_reference(set: &NettingSet, market: &Market, dates: usize) -> (CvaGreeks, Vec<f64>) {
    let step = market.time / dates as f64;
    let lgd = 1.0 - set.recovery;
    let lambda = set.hazard_rate;
    let mut reference = CvaGreeks {
        cva: 0.0,
        delta: 0.0,
        rho: 0.0,
        vegas: vec![0.0; market.vols.len()],
        hazards: vec![0.0],
    };
    let mut exposure = vec![0.0; dates];
    for trade in &set.trades {
        let (g, vegas) = closed_form(trade, market);
        let n = trade.notional;
        let last = trade.maturity - step;
        let default_probability = 1.0 - (-lambda * last).exp();
        let weight = lgd * n * default_probability;
        reference.cva += weight * g.price;
        reference.delta += weight * g.delta;
        reference.rho += weight * g.rho;
        for (r, v) in reference.vegas.iter_mut().zip(&vegas) {
            *r += weight * v;
        }
        reference.hazards[0] += lgd * n * g.price * last * (-lambda * last).exp();
        for (j, e) in exposure.iter_mut().enumerate() {
            let t = step * (j + 1) as f64;
            if t < trade.maturity - 0.5 * step {
                *e += n * g.price * (market.rate * t).exp();
            }
        }
    }
    (reference, exposure)
}

#[test]
fn long_only_cva_and_sensitivities_match_the_closed_form() {
    let long_only = NettingSet {
        trades: vec![
            LONG_CALL,
            LONG_PUT,
            Trade {
                strike: 95.0,
                ..LONG_PUT
            },
        ],
        hazard_rate: 0.03,
        recovery: RECOVERY,
    };
    let result = cva_of(std::slice::from_ref(&long_only), &MARKET);
    let set = &result.sets[0];
    let (mc, se) = (&set.greeks, &set.std_error);
    let (reference, exposure) = long_only_reference(&long_only, &MARKET, EXPOSURE_DATES);

    assert_within_mc_tolerance("cva", mc.cva, reference.cva, se.cva);
    assert_within_mc_tolerance("delta", mc.delta, reference.delta, se.delta);
    assert_within_mc_tolerance("rho", mc.rho, reference.rho, se.rho);
    for k in 0..VOLS.len() {
        assert_within_mc_tolerance(
            &format!("bucket {k} vega"),
            mc.vegas[k],
            reference.vegas[k],
            se.vegas[k],
        );
    }
    assert_within_mc_tolerance("hazard", mc.hazards[0], reference.hazards[0], se.hazards[0]);
    assert_eq!(set.expected_exposure.len(), EXPOSURE_DATES);
    let profile = set
        .expected_exposure
        .iter()
        .zip(&set.expected_exposure_std_error)
        .zip(&exposure);
    for (j, ((&ee, &ee_se), &reference)) in profile.enumerate() {
        let name = format!("expected exposure at date {}", j + 1);
        if reference == 0.0 {
            // Every trade has matured: no exposure on any path.
            assert_eq!((ee, ee_se), (0.0, 0.0), "{name}");
        } else {
            assert_within_mc_tolerance(&name, ee, reference, ee_se);
        }
    }
}

/// Four equal buckets of 0.5y on 8 quarterly dates over 2y: every bucket
/// edge is an exposure date, so the grid and the normals are the flat ones.
#[test]
fn equal_bucket_vegas_sum_to_the_flat_vega_on_common_paths() {
    let sets = [set(&[LONG_CALL, SHORT_PUT, LONG_PUT])];
    let flat_vols = [0.2];
    let curve_vols = [0.2; 4];
    let flat = cva_of(
        &sets,
        &Market {
            vols: &flat_vols,
            ..MARKET
        },
    );
    let curve = cva_of(
        &sets,
        &Market {
            vols: &curve_vols,
            ..MARKET
        },
    );
    let (f, c) = (&flat.sets[0].greeks, &curve.sets[0].greeks);

    let sum: f64 = c.vegas.iter().sum();
    assert!(
        (sum - f.vegas[0]).abs() < 1e-10 * f.vegas[0].abs(),
        "sum of bucket vegas {sum} vs flat Vega {}",
        f.vegas[0]
    );
    assert!(
        (c.cva - f.cva).abs() < 1e-12 * f.cva,
        "cva {} vs {}",
        c.cva,
        f.cva
    );
    assert!((c.delta - f.delta).abs() < 1e-10 * f.delta.abs(), "delta");
    assert!((c.rho - f.rho).abs() < 1e-10 * f.rho.abs(), "rho");
}

/// Three counterparties with different hazard rates on common paths.
#[test]
fn cross_set_hazard_sensitivities_are_zero_and_the_total_is_the_sum() {
    let sets = [
        NettingSet {
            hazard_rate: 0.01,
            ..set(&[LONG_CALL, SHORT_PUT])
        },
        NettingSet {
            hazard_rate: 0.02,
            ..set(&[LONG_PUT])
        },
        NettingSet {
            hazard_rate: 0.03,
            ..set(&[
                LONG_CALL,
                Trade {
                    notional: -0.5,
                    ..LONG_PUT
                },
            ])
        },
    ];
    let result = cva_of(&sets, &MARKET);

    for (i, s) in result.sets.iter().enumerate() {
        assert_eq!(s.greeks.hazards.len(), sets.len());
        assert!(s.greeks.hazards[i] > 0.0, "set {i} own hazard sensitivity");
        for l in (0..sets.len()).filter(|&l| l != i) {
            assert_eq!(s.greeks.hazards[l], 0.0, "set {i} wrt hazard {l}");
            assert_eq!(s.std_error.hazards[l], 0.0, "set {i} wrt hazard {l} SE");
        }
        // The total's sensitivity to hazard i is set i's on every path.
        assert_eq!(
            result.total.hazards[i], s.greeks.hazards[i],
            "total hazard {i}"
        );
    }

    // Mean of per-path sums against the sum of means: equal up to rounding.
    let close = |name: &str, total: f64, parts: f64| {
        assert!(
            (total - parts).abs() <= 1e-12 * parts.abs().max(1e-12),
            "{name}: total {total} vs sum of sets {parts}"
        );
    };
    let sum = |f: fn(&CvaGreeks) -> f64| result.sets.iter().map(|s| f(&s.greeks)).sum::<f64>();
    close("cva", result.total.cva, sum(|g| g.cva));
    close("delta", result.total.delta, sum(|g| g.delta));
    close("rho", result.total.rho, sum(|g| g.rho));
    for k in 0..VOLS.len() {
        let parts: f64 = result.sets.iter().map(|s| s.greeks.vegas[k]).sum();
        close(&format!("bucket {k} vega"), result.total.vegas[k], parts);
    }
}

#[test]
fn tape_free_cva_reproduces_the_adjoint_cva_exactly() {
    let sets = [
        set(&[LONG_CALL, SHORT_PUT, LONG_PUT]),
        NettingSet {
            hazard_rate: 0.03,
            ..set(&[LONG_PUT])
        },
    ];
    let adjoint = cva_of(&sets, &MARKET);
    let mut rng = StdRng::seed_from_u64(SEED);
    let plain = cva_value(&sets, &MARKET, EXPOSURE_DATES, NUM_PATHS, &mut rng);
    assert_eq!(plain.len(), sets.len());
    for (i, s) in adjoint.sets.iter().enumerate() {
        assert_eq!(plain[i], s.greeks.cva, "set {i}");
    }
}

#[test]
fn mixed_set_sensitivities_match_bump_and_reval() {
    let sets = [
        set(&[
            LONG_CALL,
            SHORT_PUT,
            LONG_PUT,
            Trade {
                kind: OptionKind::Call,
                strike: 110.0,
                maturity: 1.5,
                notional: -1.0,
            },
        ]),
        NettingSet {
            hazard_rate: 0.01,
            ..set(&[
                Trade {
                    notional: -1.0,
                    ..LONG_CALL
                },
                Trade {
                    strike: 105.0,
                    maturity: 0.5,
                    ..LONG_PUT
                },
            ])
        },
    ];
    let adjoint = cva_of(&sets, &MARKET);
    let bumped = bump_and_reval_cva(&sets, &MARKET, EXPOSURE_DATES, NUM_PATHS, SEED);
    assert_eq!(bumped.len(), sets.len());

    for (i, (aad, fd)) in adjoint.sets.iter().zip(&bumped).enumerate() {
        let (mc, se) = (&aad.greeks, &aad.std_error);
        let name = |q: &str| format!("set {i}: {q}");
        assert!(mc.cva > 0.0, "{}", name("cva"));
        assert_eq!(fd.cva, mc.cva, "{}", name("cva"));
        assert_within_mc_tolerance(&name("delta"), mc.delta, fd.delta, se.delta);
        assert_within_mc_tolerance(&name("rho"), mc.rho, fd.rho, se.rho);
        for k in 0..VOLS.len() {
            assert_within_mc_tolerance(
                &name(&format!("bucket {k} vega")),
                mc.vegas[k],
                fd.vegas[k],
                se.vegas[k],
            );
        }
        for l in 0..sets.len() {
            if l == i {
                assert_within_mc_tolerance(
                    &name("own hazard"),
                    mc.hazards[l],
                    fd.hazards[l],
                    se.hazards[l],
                );
            } else {
                // Bumping another counterparty's hazard rate leaves this set
                // untouched on every path, so both methods give exactly zero.
                assert_eq!(
                    (mc.hazards[l], fd.hazards[l]),
                    (0.0, 0.0),
                    "{}",
                    name("other hazard")
                );
            }
        }
    }
}
