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
        }
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

/// Record an independent variable and return its tape index.
pub(crate) fn push_leaf() -> u32 {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        let index = tape.nodes.len() as u32;
        tape.push(Node {
            parents: [index, index],
            partials: [0.0, 0.0],
        })
    })
}

/// Record a one-input operation given its local partial derivative.
pub(crate) fn push_unary(parent: u32, partial: f64) -> u32 {
    TAPE.with(|t| {
        t.borrow_mut().push(Node {
            parents: [parent, parent],
            partials: [partial, 0.0],
        })
    })
}

/// Record a two-input operation given both local partial derivatives.
pub(crate) fn push_binary(left: u32, d_left: f64, right: u32, d_right: f64) -> u32 {
    TAPE.with(|t| {
        t.borrow_mut().push(Node {
            parents: [left, right],
            partials: [d_left, d_right],
        })
    })
}

/// Seed the adjoint of `output` at 1.0 and propagate it back through every
/// node recorded before it.
///
/// We start the sweep at `output` rather than the end of the tape, because
/// anything recorded after `output` couldn't possibly have influenced it —
/// there's no point walking through it.
pub(crate) fn run_backward(output: u32) {
    TAPE.with(|t| {
        let tape = &mut *t.borrow_mut();
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
    });
}

/// Look up the adjoint a node ended up with after a backward sweep.
///
/// If a node was recorded after the sweep's output, it couldn't have
/// influenced it — so its sensitivity is just zero.
pub(crate) fn adjoint_of(index: u32) -> f64 {
    TAPE.with(|t| {
        t.borrow()
            .adjoints
            .get(index as usize)
            .copied()
            .unwrap_or(0.0)
    })
}

/// Throw away everything we've recorded so far, but hang onto the arena's
/// memory so we can reuse it.
///
/// This is what runs between Monte Carlo paths — it's just an O(1) pointer
/// reset. The chunk we already grew to stays mapped (its high-water mark),
/// so every path after the first one can bump-allocate its nodes without
/// touching the heap at all.
pub fn reset() {
    TAPE.with(|t| {
        let tape = &mut *t.borrow_mut();
        tape.nodes.clear();
        tape.adjoints.clear();
    });
}

/// Number of operations currently recorded.
pub fn len() -> usize {
    TAPE.with(|t| t.borrow().nodes.len())
}
