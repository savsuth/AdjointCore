//! Handle validity: a `Dual` is only good on the tape recording it came
//! from, and a `Gradient` only until the next sweep or reset. Using either
//! past that point has to fail loudly instead of quietly reading whatever
//! now sits at the same tape index.

use std::thread;

use adjoint_core::{Dual, tape};

const TOL: f64 = 1e-12;

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// Several outputs swept one after another on the same recording are all
/// fine, as long as each gradient is read before the next sweep.
#[test]
fn current_handles_keep_working_across_sweeps() {
    tape::reset();
    let x = Dual::var(2.0);
    let z = x * x;
    let w = x * 3.0;

    let grad_z = z.backward();
    assert!((grad_z.wrt(x) - 4.0).abs() < TOL);

    let grad_w = w.backward();
    assert!((grad_w.wrt(x) - 3.0).abs() < TOL);

    // x survives both sweeps and can still feed new operations.
    let grad_v = (x + 1.0).backward();
    assert!((grad_v.wrt(x) - 1.0).abs() < TOL);
}

#[test]
#[should_panic(expected = "Gradient is stale")]
fn gradient_rejected_after_a_second_sweep() {
    tape::reset();
    let x = Dual::var(2.0);
    let first = (x * x).backward();
    let _second = (x * 3.0).backward();

    first.wrt(x);
}

#[test]
#[should_panic(expected = "Gradient is stale")]
fn gradient_rejected_after_reset() {
    tape::reset();
    let x = Dual::var(2.0);
    let grad = (x * x).backward();
    tape::reset();

    grad.wrt(x);
}

/// After a reset the new recording reuses index 0, so the old handle points
/// at the same slot as `y`. It must not read `y`'s adjoint.
#[test]
#[should_panic(expected = "Dual is stale")]
fn old_dual_rejected_by_gradient_after_reset() {
    tape::reset();
    let old_x = Dual::var(2.0);
    tape::reset();

    let y = Dual::var(10.0);
    let grad = (y * y).backward();
    grad.wrt(old_x);
}

#[test]
#[should_panic(expected = "Dual is stale")]
fn old_dual_rejected_as_operand_after_reset() {
    tape::reset();
    let old_x = Dual::var(2.0);
    tape::reset();

    let _y = Dual::var(10.0);
    let _ = old_x * 5.0;
}

#[test]
#[should_panic(expected = "Dual is stale")]
fn old_dual_rejected_as_sweep_output_after_reset() {
    tape::reset();
    let old_z = Dual::var(2.0) * 3.0;
    tape::reset();

    let _y = Dual::var(10.0);
    old_z.backward();
}

/// `Dual` is `Copy + Send`, so nothing stops it being moved to another
/// thread. That thread has its own tape, so the handle has to be refused
/// there rather than aliasing whatever that tape holds at the same index.
#[test]
fn dual_rejected_on_another_threads_tape() {
    tape::reset();
    let x = Dual::var(2.0);

    let result = thread::spawn(move || {
        let _local = Dual::var(1.0);
        let _ = x * 2.0;
    })
    .join();

    let message = panic_message(result.expect_err("foreign Dual was accepted"));
    assert!(message.contains("Dual is stale"), "message: {message}");
}

#[test]
fn gradient_rejected_on_another_thread() {
    tape::reset();
    let x = Dual::var(2.0);
    let grad = (x * x).backward();

    let result = thread::spawn(move || {
        let _local = (Dual::var(1.0) * 1.0).backward();
        grad.wrt(x)
    })
    .join();

    let message = panic_message(result.expect_err("foreign Gradient was accepted"));
    assert!(message.contains("Gradient is stale"), "message: {message}");
}
