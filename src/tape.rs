//! Arena-backed adjoint tape for reverse-mode sensitivity calculation.
//!
//! The tape is basically a flat log we keep appending to as we value
//! something forward. Every entry records the local partial derivatives of
//! that one operation with respect to its inputs, and then the backward
//! sweep walks the log in reverse, applying the chain rule as it goes to
//! build up the adjoints.
//!
//! We keep one tape per thread. Monte Carlo paths get differentiated
//! path-wise (∇E[V] = E[∇V]) — so for each path we record it, sweep it,
//! bank whatever adjoints came out, and reset before starting the next one.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

/// How much we reserve for the arena up front, in bytes. Sized so a typical
/// single-path tape never has to grow past this first chunk and just stays
/// resident in L1/L2.
const ARENA_BYTES: usize = 64 * 1024;

/// A single recorded operation.
///
/// We use `u32` for the indices instead of `usize` — that packs a node into
/// 24 bytes instead of 32, so a third more of the tape fits in cache during
/// the backward sweep.
///
/// A leaf (an independent variable, like Spot or Vol) just points both
/// parent slots back at itself with zero partials. Propagating into yourself
/// adds nothing, so the sweep doesn't need a special branch to recognize
/// leaves — it just falls out naturally.
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub parents: [u32; 2],
    pub partials: [f64; 2],
}

pub struct Tape {
    nodes: BumpVec<'static, Node>,
    /// Where adjoints accumulate. We reuse this buffer across sweeps so
    /// repeated pricing never has to go back to the system allocator.
    adjoints: Vec<f64>,
    /// Which recording this is. Every `Dual` is stamped with the generation
    /// it was recorded under, and anything that hands an index back to the
    /// tape has to match it. A reset moves to a fresh generation, and the
    /// ids come from one process-wide counter, so a handle from before a
    /// reset or from another thread's tape never matches.
    generation: u32,
    /// Counts backward sweeps, so a `Gradient` can tell whether the adjoint
    /// buffer still holds the sweep it was handed.
    sweep: u32,
}

/// Source of tape generation ids, shared by every thread's tape.
///
/// A `u32` keeps `Dual` at 16 bytes (the id sits in what used to be
/// padding). It wraps after 2^32 resets process-wide, so a stale handle
/// would only slip through if it was used exactly a multiple of 2^32
/// generations later.
static NEXT_GENERATION: AtomicU32 = AtomicU32::new(0);

fn next_generation() -> u32 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

impl Tape {
    fn new() -> Self {
        // We leak the arena to get the `'static` lifetime BumpVec wants —
        // that sidesteps a self-referential struct entirely. It's fine to do
        // this because the arena belongs to a thread-local anyway, so it
        // would've lived for the whole thread's lifetime regardless.
        let arena: &'static Bump = Box::leak(Box::new(Bump::with_capacity(ARENA_BYTES)));
        Tape {
            nodes: BumpVec::new_in(arena),
            adjoints: Vec::new(),
            generation: next_generation(),
            sweep: 0,
        }
    }

    /// Refuse a `Dual` that wasn't recorded on this tape since its last
    /// reset. Its index would otherwise point at an unrelated node.
    fn check(&self, generation: u32) {
        assert!(
            generation == self.generation,
            "Dual is stale: it was recorded before the last tape::reset() or on another \
             thread's tape, so its index no longer refers to it"
        );
    }

    fn push(&mut self, node: Node) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(node);
        index
    }
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

/// Record an independent variable and return its tape index and the
/// generation it belongs to.
pub(crate) fn push_leaf() -> (u32, u32) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        let index = tape.nodes.len() as u32;
        let index = tape.push(Node {
            parents: [index, index],
            partials: [0.0, 0.0],
        });
        (index, tape.generation)
    })
}

/// Record a one-input operation given its local partial derivative.
pub(crate) fn push_unary(parent: (u32, u32), partial: f64) -> (u32, u32) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.check(parent.1);
        let index = tape.push(Node {
            parents: [parent.0, parent.0],
            partials: [partial, 0.0],
        });
        (index, tape.generation)
    })
}

/// Record a two-input operation given both local partial derivatives.
pub(crate) fn push_binary(
    left: (u32, u32),
    d_left: f64,
    right: (u32, u32),
    d_right: f64,
) -> (u32, u32) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.check(left.1);
        tape.check(right.1);
        let index = tape.push(Node {
            parents: [left.0, right.0],
            partials: [d_left, d_right],
        });
        (index, tape.generation)
    })
}

/// Seed the adjoint of `output` at 1.0 and propagate it back through every
/// node recorded before it.
///
/// We start the sweep at `output` rather than the end of the tape, because
/// anything recorded after `output` couldn't possibly have influenced it —
/// there's no point walking through it.
///
/// Returns the `(generation, sweep)` pair a `Gradient` needs to check later
/// that these adjoints haven't been overwritten.
pub(crate) fn run_backward(output: (u32, u32)) -> (u32, u32) {
    TAPE.with(|t| {
        let tape = &mut *t.borrow_mut();
        tape.check(output.1);
        tape.sweep = tape.sweep.wrapping_add(1);
        let output = output.0;
        let len = output as usize + 1;

        tape.adjoints.clear();
        tape.adjoints.resize(len, 0.0);
        tape.adjoints[output as usize] = 1.0;

        for i in (0..len).rev() {
            let adjoint = tape.adjoints[i];
            if adjoint == 0.0 {
                continue;
            }
            let node = tape.nodes[i];
            tape.adjoints[node.parents[0] as usize] += adjoint * node.partials[0];
            tape.adjoints[node.parents[1] as usize] += adjoint * node.partials[1];
        }
        (tape.generation, tape.sweep)
    })
}

/// Look up the adjoint a node ended up with after a backward sweep.
///
/// If a node was recorded after the sweep's output, it couldn't have
/// influenced it — so its sensitivity is just zero.
///
/// `sweep` is what `run_backward` returned; if another sweep or a reset has
/// happened since, the buffer holds someone else's adjoints and we refuse.
pub(crate) fn adjoint_of(sweep: (u32, u32), input: (u32, u32)) -> f64 {
    TAPE.with(|t| {
        let tape = t.borrow();
        assert!(
            sweep == (tape.generation, tape.sweep),
            "Gradient is stale: backward() or tape::reset() has run since it was \
             produced (or it was moved to another thread), so read every derivative \
             you need before the next sweep"
        );
        tape.check(input.1);
        tape.adjoints.get(input.0 as usize).copied().unwrap_or(0.0)
    })
}

/// Throw away everything we've recorded so far, but hang onto the arena's
/// memory so we can reuse it.
///
/// This is what runs between Monte Carlo paths — it's just an O(1) pointer
/// reset. The chunk we already grew to stays mapped (its high-water mark),
/// so every path after the first one can bump-allocate its nodes without
/// touching the heap at all.
///
/// It also moves the tape to a new generation, which invalidates every
/// `Dual` and `Gradient` from before the reset (one relaxed atomic add).
pub fn reset() {
    TAPE.with(|t| {
        let tape = &mut *t.borrow_mut();
        tape.nodes.clear();
        tape.adjoints.clear();
        tape.generation = next_generation();
    });
}

/// Number of operations currently recorded.
pub fn len() -> usize {
    TAPE.with(|t| t.borrow().nodes.len())
}
