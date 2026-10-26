//! Unilateral CVA of netting sets of European options, and every
//! sensitivity of it from one recording per path.
//!
//! A netting set is the trades facing one counterparty. If the counterparty
//! defaults, the trades are closed out at their netted value, and the loss is
//! the positive part of that value times (1 - R). On exposure dates
//! t_j = j T / m, with T the longest maturity,
//!
//! ```text
//! CVA = (1 - R) sum_j E[ D(t_j) max(V(t_j), 0) ] (e^{-lambda t_{j-1}} - e^{-lambda t_j})
//! ```
//!
//! with D the discount factor, V the netted value, and a flat hazard rate
//! lambda per counterparty, independent of the spot. Netting is what makes
//! this nonlinear: max(sum v_i, 0) <= sum max(v_i, 0), so a set's CVA is at
//! most the sum of its trades' standalone CVAs.
//!
//! The value of each trade at an exposure date is the Black-Scholes closed
//! form on the remaining integrated variance of the vol curve, exact for
//! piecewise-constant vol, written once against [`Real`]. On `Dual`, spot,
//! rate, every bucket vol and every set's hazard rate are tape variables, so
//! they reach the CVA through the simulated spot, the future values, the
//! discount factors and the default probabilities. Each path is recorded
//! once, then swept once per netting set: the cost grows with the number of
//! sets, not with the number of inputs. max(V, 0) is Lipschitz in the path,
//! so the pathwise estimator is unbiased.

use rand::Rng;

use crate::dual::Dual;
use crate::payoff::Real;
use crate::pricer::{Market, RunningStats, TimeGrid, draw_normals, simulate_fixings, time_grid};
use crate::tape;

/// Call or put.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Call,
    Put,
}

/// A European option on the single underlying. `notional` is signed:
/// positive when we hold the option, negative when we sold it. `maturity`
/// must fall on an exposure date.
#[derive(Clone, Copy, Debug)]
pub struct Trade {
    pub kind: OptionKind,
    pub strike: f64,
    pub maturity: f64,
    pub notional: f64,
}

/// The trades facing one counterparty, with its flat hazard rate and its
/// recovery rate.
#[derive(Clone, Debug)]
pub struct NettingSet {
    pub trades: Vec<Trade>,
    pub hazard_rate: f64,
    pub recovery: f64,
}

/// A CVA and its sensitivities: Delta, Rho, one vega per vol bucket, and
/// one sensitivity per netting set's hazard rate, in set order.
#[derive(Clone, Debug, PartialEq)]
pub struct CvaGreeks {
    pub cva: f64,
    pub delta: f64,
    pub rho: f64,
    pub vegas: Vec<f64>,
    pub hazards: Vec<f64>,
}

/// One netting set's CVA estimate with its standard errors, and its
/// expected exposure profile E[max(V(t_j), 0)] (undiscounted) at each
/// exposure date, with standard errors.
#[derive(Clone, Debug, PartialEq)]
pub struct NettingSetResult {
    pub greeks: CvaGreeks,
    pub std_error: CvaGreeks,
    pub expected_exposure: Vec<f64>,
    pub expected_exposure_std_error: Vec<f64>,
}

/// Every netting set's result, in set order, and the total across sets
/// with its own standard errors (from the per-path totals).
#[derive(Clone, Debug, PartialEq)]
pub struct CvaResult {
    pub sets: Vec<NettingSetResult>,
    pub total: CvaGreeks,
    pub total_std_error: CvaGreeks,
}

impl CvaGreeks {
    fn zero(buckets: usize, sets: usize) -> Self {
        CvaGreeks {
            cva: 0.0,
            delta: 0.0,
            rho: 0.0,
            vegas: vec![0.0; buckets],
            hazards: vec![0.0; sets],
        }
    }

    fn clear(&mut self) {
        self.cva = 0.0;
        self.delta = 0.0;
        self.rho = 0.0;
        self.vegas.fill(0.0);
        self.hazards.fill(0.0);
    }

    fn add(&mut self, other: &CvaGreeks) {
        self.cva += other.cva;
        self.delta += other.delta;
        self.rho += other.rho;
        self.vegas
            .iter_mut()
            .zip(&other.vegas)
            .for_each(|(a, b)| *a += b);
        self.hazards
            .iter_mut()
            .zip(&other.hazards)
            .for_each(|(a, b)| *a += b);
    }
}

/// The exposure dates on the simulation grid, and where each trade
/// matures on them.
struct Schedule {
    grid: TimeGrid,
    /// t_1, ..., t_m.
    dates: Vec<f64>,
    /// Per set, per trade: the exposure date number (1..=m) of its maturity.
    maturities: Vec<Vec<usize>>,
}

fn schedule(sets: &[NettingSet], market: &Market, exposure_dates: usize) -> Schedule {
    assert!(exposure_dates > 0, "CVA needs at least one exposure date");
    let horizon = market.time;
    let longest = sets
        .iter()
        .flat_map(|s| &s.trades)
        .map(|t| t.maturity)
        .fold(0.0, f64::max);
    assert!(
        (longest - horizon).abs() <= 1e-12 * horizon,
        "market.time ({horizon}) must be the longest maturity ({longest})"
    );

    let m = exposure_dates;
    let maturities = sets
        .iter()
        .map(|s| {
            s.trades
                .iter()
                .map(|t| {
                    let q = (t.maturity / horizon * m as f64).round() as usize;
                    let on_grid = horizon * q as f64 / m as f64;
                    assert!(
                        q >= 1 && (on_grid - t.maturity).abs() <= 1e-12 * horizon,
                        "maturity {} is not an exposure date (multiples of {})",
                        t.maturity,
                        horizon / m as f64
                    );
                    q
                })
                .collect()
        })
        .collect();

    Schedule {
        grid: time_grid(m, market.vols.len(), horizon),
        dates: (1..=m).map(|j| horizon * j as f64 / m as f64).collect(),
        maturities,
    }
}

/// What every netting set reads from one path: the spot, its log, the
/// integrated variance from 0, and the discount factor at each exposure
/// date. Buffers are reused across paths.
struct PathState<T> {
    spots: Vec<T>,
    ln_spots: Vec<T>,
    variance: Vec<T>,
    discounts: Vec<T>,
    vol_squares: Vec<T>,
}

impl<T: Real> PathState<T> {
    fn new(dates: usize, buckets: usize) -> Self {
        PathState {
            spots: Vec::with_capacity(dates),
            ln_spots: Vec::with_capacity(dates),
            variance: Vec::with_capacity(dates),
            discounts: Vec::with_capacity(dates),
            vol_squares: Vec::with_capacity(buckets),
        }
    }

    fn simulate(&mut self, schedule: &Schedule, spot: T, rate: T, vols: &[T], normals: &[f64]) {
        simulate_fixings(&schedule.grid, spot, rate, vols, normals, &mut self.spots);
        self.ln_spots.clear();
        self.ln_spots.extend(self.spots.iter().map(|&s| s.ln()));

        // Integrated variance on the merged grid, read off at each date.
        self.vol_squares.clear();
        self.vol_squares.extend(vols.iter().map(|&v| v * v));
        self.variance.clear();
        let mut total: Option<T> = None;
        for step in &schedule.grid.steps {
            let piece = self.vol_squares[step.bucket] * step.dt;
            let sum = match total {
                Some(sum) => sum + piece,
                None => piece,
            };
            total = Some(sum);
            if step.fixing {
                self.variance.push(sum);
            }
        }

        self.discounts.clear();
        self.discounts
            .extend(schedule.dates.iter().map(|&t| (rate * -t).exp()));
    }
}

/// The Black-Scholes value at exposure date number `date` (1-based) of a
/// trade maturing on date number `maturity` > `date`, on the variance left
/// between the two.
fn future_value<T: Real>(
    trade: &Trade,
    date: usize,
    maturity: usize,
    rate: T,
    schedule: &Schedule,
    state: &PathState<T>,
) -> T {
    let (d, q) = (date - 1, maturity - 1);
    let tau = schedule.dates[q] - schedule.dates[d];
    let variance = state.variance[q] - state.variance[d];
    let sd = variance.sqrt();
    let d1 = (state.ln_spots[d] - trade.strike.ln() + rate * tau + variance * 0.5) / sd;
    let d2 = d1 - sd;
    let discounted_strike = (rate * -tau).exp() * trade.strike;
    let spot = state.spots[d];
    match trade.kind {
        OptionKind::Call => spot * d1.norm_cdf() - discounted_strike * d2.norm_cdf(),
        OptionKind::Put => discounted_strike * (-d2).norm_cdf() - spot * (-d1).norm_cdf(),
    }
}

/// One path's CVA of one netting set, and its exposure max(V(t_j), 0) at
/// each date (`None` where no trade is alive). `None` overall when no trade
/// is alive on any date: the CVA is then exactly zero.
fn set_cva<T: Real>(
    set: &NettingSet,
    maturities: &[usize],
    hazard: T,
    rate: T,
    schedule: &Schedule,
    state: &PathState<T>,
    exposures: &mut Vec<Option<T>>,
) -> Option<T> {
    exposures.clear();
    let mut cva: Option<T> = None;
    let mut survival_before = (hazard * -0.0).exp();
    for (d, &t) in schedule.dates.iter().enumerate() {
        let date = d + 1;
        let survival = (hazard * -t).exp();
        let value = set
            .trades
            .iter()
            .zip(maturities)
            .filter(|&(_, &q)| q > date)
            .map(|(trade, &q)| future_value(trade, date, q, rate, schedule, state) * trade.notional)
            .reduce(|a, b| a + b);
        let exposure = value.map(|v| v.max_zero());
        exposures.push(exposure);
        if let Some(e) = exposure {
            let term = state.discounts[d] * e * (survival_before - survival);
            cva = Some(match cva {
                Some(sum) => sum + term,
                None => term,
            });
        }
        survival_before = survival;
    }
    cva.map(|c| c * (1.0 - set.recovery))
}

/// Running statistics of one CVA estimator and its sensitivities.
struct CvaStats {
    cva: RunningStats,
    delta: RunningStats,
    rho: RunningStats,
    vegas: Vec<RunningStats>,
    hazards: Vec<RunningStats>,
}

impl CvaStats {
    fn new(buckets: usize, sets: usize) -> Self {
        CvaStats {
            cva: RunningStats::default(),
            delta: RunningStats::default(),
            rho: RunningStats::default(),
            vegas: vec![RunningStats::default(); buckets],
            hazards: vec![RunningStats::default(); sets],
        }
    }

    fn update(&mut self, x: &CvaGreeks) {
        self.cva.update(x.cva);
        self.delta.update(x.delta);
        self.rho.update(x.rho);
        self.vegas
            .iter_mut()
            .zip(&x.vegas)
            .for_each(|(s, &v)| s.update(v));
        self.hazards
            .iter_mut()
            .zip(&x.hazards)
            .for_each(|(s, &h)| s.update(h));
    }

    fn means(&self) -> CvaGreeks {
        CvaGreeks {
            cva: self.cva.mean,
            delta: self.delta.mean,
            rho: self.rho.mean,
            vegas: self.vegas.iter().map(|s| s.mean).collect(),
            hazards: self.hazards.iter().map(|s| s.mean).collect(),
        }
    }

    fn standard_errors(&self) -> CvaGreeks {
        CvaGreeks {
            cva: self.cva.standard_error(),
            delta: self.delta.standard_error(),
            rho: self.rho.standard_error(),
            vegas: self.vegas.iter().map(|s| s.standard_error()).collect(),
            hazards: self.hazards.iter().map(|s| s.standard_error()).collect(),
        }
    }
}

/// Monte Carlo CVA of each netting set and of their total, with Delta,
/// Rho, one vega per bucket of `market.vols`, and one sensitivity per
/// set's hazard rate.
///
/// `exposure_dates` (m) equally spaced dates end at `market.time`, which
/// must be the longest maturity; every maturity must be one of the dates,
/// and a trade counts on the dates strictly before it. Each path records
/// spot, rate, every bucket vol and every hazard rate on the tape once,
/// prices every set on it, and sweeps once per set.
pub fn monte_carlo_cva<R: Rng>(
    sets: &[NettingSet],
    market: &Market,
    exposure_dates: usize,
    num_paths: u64,
    rng: &mut R,
) -> CvaResult {
    let schedule = schedule(sets, market, exposure_dates);
    let (buckets, k) = (market.vols.len(), sets.len());

    let mut stats: Vec<CvaStats> = (0..k).map(|_| CvaStats::new(buckets, k)).collect();
    let mut exposure_stats = vec![vec![RunningStats::default(); exposure_dates]; k];
    let mut total_stats = CvaStats::new(buckets, k);

    let mut state = PathState::new(exposure_dates, buckets);
    let mut normals = vec![0.0; schedule.grid.steps.len()];
    let mut vol_vars = Vec::with_capacity(buckets);
    let mut hazard_vars = Vec::with_capacity(k);
    let mut exposures = Vec::with_capacity(exposure_dates);
    let mut cvas: Vec<Option<Dual>> = Vec::with_capacity(k);
    let mut sample = CvaGreeks::zero(buckets, k);
    let mut total = CvaGreeks::zero(buckets, k);

    for _ in 0..num_paths {
        let s = Dual::var(market.spot);
        let r = Dual::var(market.rate);
        vol_vars.clear();
        vol_vars.extend(market.vols.iter().map(|&v| Dual::var(v)));
        hazard_vars.clear();
        hazard_vars.extend(sets.iter().map(|set| Dual::var(set.hazard_rate)));

        draw_normals(&mut normals, rng);
        state.simulate(&schedule, s, r, &vol_vars, &normals);

        cvas.clear();
        for (i, set) in sets.iter().enumerate() {
            let cva = set_cva(
                set,
                &schedule.maturities[i],
                hazard_vars[i],
                r,
                &schedule,
                &state,
                &mut exposures,
            );
            for (stat, e) in exposure_stats[i].iter_mut().zip(&exposures) {
                stat.update(e.map_or(0.0, |e| e.value));
            }
            cvas.push(cva);
        }

        total.clear();
        for (i, cva) in cvas.iter().enumerate() {
            sample.clear();
            if let Some(cva) = cva {
                let grad = cva.backward();
                sample.cva = cva.value;
                sample.delta = grad.wrt(s);
                sample.rho = grad.wrt(r);
                for (vega, &vol) in sample.vegas.iter_mut().zip(&vol_vars) {
                    *vega = grad.wrt(vol);
                }
                for (h, &hazard) in sample.hazards.iter_mut().zip(&hazard_vars) {
                    *h = grad.wrt(hazard);
                }
            }
            stats[i].update(&sample);
            total.add(&sample);
        }
        total_stats.update(&total);

        tape::reset();
    }

    let sets = stats
        .iter()
        .zip(&exposure_stats)
        .map(|(s, e)| NettingSetResult {
            greeks: s.means(),
            std_error: s.standard_errors(),
            expected_exposure: e.iter().map(|x| x.mean).collect(),
            expected_exposure_std_error: e.iter().map(|x| x.standard_error()).collect(),
        })
        .collect();
    CvaResult {
        sets,
        total: total_stats.means(),
        total_std_error: total_stats.standard_errors(),
    }
}

/// Monte Carlo CVA of each netting set with nothing recorded on the tape.
///
/// Same paths as [`monte_carlo_cva`] for the same seed, the same `f64`
/// arithmetic and the same accumulator, so each set's CVA matches that one
/// exactly. This is the revaluation that [`crate::bump_and_reval_cva`]
/// repeats once per bumped input.
pub fn cva_value<R: Rng>(
    sets: &[NettingSet],
    market: &Market,
    exposure_dates: usize,
    num_paths: u64,
    rng: &mut R,
) -> Vec<f64> {
    let schedule = schedule(sets, market, exposure_dates);
    let mut stats = vec![RunningStats::default(); sets.len()];
    let mut state = PathState::new(exposure_dates, market.vols.len());
    let mut normals = vec![0.0; schedule.grid.steps.len()];
    let mut exposures = Vec::with_capacity(exposure_dates);

    for _ in 0..num_paths {
        draw_normals(&mut normals, rng);
        state.simulate(&schedule, market.spot, market.rate, market.vols, &normals);
        for (i, set) in sets.iter().enumerate() {
            let cva = set_cva(
                set,
                &schedule.maturities[i],
                set.hazard_rate,
                market.rate,
                &schedule,
                &state,
                &mut exposures,
            );
            stats[i].update(cva.unwrap_or(0.0));
        }
    }

    stats.iter().map(|s| s.mean).collect()
}
