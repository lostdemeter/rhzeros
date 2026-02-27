#!/usr/bin/env python3
"""
Geometric Zeta Zero Hunter
===========================

A purely geometric approach to finding Riemann zeta zeros by index.
No Gram points. No sweep. Direct: index n → zero t_n.

Pipeline:
---------
  Stage 1 (Compressor): Lambert W inversion of the smooth counting
                        function → O(1) global estimate.
  Stage 2 (Processor):  Ramanujan refinement — Newton iteration on
                        the exact θ(T)/π + 1 = n to pin the smooth
                        coordinate to machine precision.
  Stage 3 (Targeter):   Riemann-Siegel Z(t) evaluation, local zero
                        search with sequential indexing, bisection,
                        and Newton polish.

Mathematical Foundation:
------------------------
  - Exact Riemann-Siegel θ(t) via scipy.special.loggamma (not the
    asymptotic Stirling expansion, which diverges for t < 20).
  - Smooth counting function N(T) = θ(T)/π + 1 as the coordinate
    system — zeros are indexed by their position in this geometry.
  - Riemann-Siegel Z(t) = real-valued function whose sign changes
    are the zeta zeros on the critical line.
  - Sequential indexing: sort candidate zeros by position, compute
    base index from median N_smooth + 0.5 bias correction, select
    by ordering. This handles close zero pairs and avoids Gram
    point dependence entirely.

Performance:
------------
  - 100/100 index-accurate (verified against mpmath.zetazero)
  - Mean |error| ~ 2e-4 (limited by R-S C₀-only remainder)
  - ~6ms per zero at n=100, ~60ms at n=10000
  - No mpmath dependency for zero-finding (scipy/numpy only)

The Expanding Tensor:
---------------------
  The R-S sum has N(t) = floor(√(t/2π)) terms. As t grows, the
  tensor gains rotation axes. Each term n^{-1/2}·cos(θ - t·ln(n))
  is a rotation. A zero is where all rotations cancel.

  This IS the ideal transformer: terms = tokens, phases = position
  encodings, amplitudes = embeddings, zero = output.

Usage:
------
    from geometric_zetas import zetazero, zetazero_batch

    # Single zero (returns float, ~1e-4 precision)
    t = zetazero(100)

    # Batch of zeros
    zeros = zetazero_batch(1, 100)

    # High-precision (uses mpmath Newton polish)
    t = zetazero(100, high_precision=True, dps=50)

References:
-----------
  - DC 048: The Curved Arithmetic Axis
  - DC 271: The Expanding Tensor
  - F107-113: Geometric zeta analysis findings
  - Doc 270: ζ IS the Ideal Transformer
"""

import numpy as np
from scipy.special import lambertw, loggamma
from functools import lru_cache
import time


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

TWO_PI = 2 * np.pi

# First 30 known zeros (for verification)
KNOWN_ZEROS = [
    14.134725141734694, 21.022039638771555, 25.010857580145689,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147500, 43.327073280914999, 48.005150881167160,
    49.773832477672302, 52.970321477714460, 56.446247697063394,
    59.347044002602353, 60.831778524609809, 65.112544048081607,
    67.079810529494174, 69.546401711173980, 72.067157674481908,
    75.704690699083933, 77.144840068874805, 79.337375020249367,
    82.910380854086030, 84.735492980517050, 87.425274613125196,
    88.809111207634465, 92.491899270228280, 94.651344040519838,
    95.870634228245309, 98.831194218193692, 101.31785100573139,
]


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: COMPRESSOR — Lambert W estimate
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=50000)
def lambert_w_estimate(n):
    """Lambert W inversion of the smooth counting function.

    Solves N_smooth(T) ≈ n using the leading-order approximation:
        T ≈ 2π(n - 7/8) / W((n - 7/8)/e)

    This captures >95% of the answer in O(1).
    """
    shift = n - 7/8
    if shift <= 0:
        return 10.0
    w = float(np.real(lambertw(shift / np.e)))
    return TWO_PI * shift / w


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRIC PRIMITIVES — exact θ, smooth counting, Z(t)
# ═══════════════════════════════════════════════════════════════════════

def riemann_siegel_theta(t):
    """Riemann-Siegel theta function — EXACT via loggamma.

    θ(t) = Im(log Γ(1/4 + it/2)) - (t/2)log(π)

    This is the phase geometry of the critical line.
    Using loggamma gives machine precision for ALL t,
    not just the asymptotic regime (t >> 1).

    The asymptotic (Stirling) expansion diverges badly for t < 20,
    giving errors > 1 at the first zero. loggamma has no such limit.
    """
    if t < 0.01:
        return 0.0
    return float(np.imag(loggamma(0.25 + 0.5j * t))) - (t / 2) * np.log(np.pi)


def theta_derivative(t, dt=1e-8):
    """dθ/dt — exact numerical derivative."""
    return (riemann_siegel_theta(t + dt) - riemann_siegel_theta(t - dt)) / (2 * dt)


def smooth_count(T):
    """Smooth zero counting function: N_smooth(T) = θ(T)/π + 1.

    This is the global coordinate system for zeros. The nth zero
    has N_smooth(t_n) ≈ n (offset by the oscillatory part S(t)).
    """
    return riemann_siegel_theta(T) / np.pi + 1


def smooth_count_derivative(T):
    """dN_smooth/dT = θ'(T)/π — the local density of zeros."""
    return theta_derivative(T) / np.pi


def riemann_siegel_Z(t):
    """Riemann-Siegel Z function — the real-valued zeta on the critical line.

    Z(t) = exp(iθ(t)) · ζ(1/2 + it)

    This is real-valued. Its sign changes are the zeta zeros.

    Z(t) = 2 Σ_{n=1}^{N} n^{-1/2} cos(θ(t) - t·ln(n)) + R(t)

    where N = floor(√(t/2π)) and R is the remainder.

    Each term is a rotation in the expanding tensor. The number of
    terms (rotation axes) grows as √t.
    """
    if t < 1:
        return 0.0

    N = int(np.sqrt(t / TWO_PI))
    if N < 1:
        N = 1

    theta = riemann_siegel_theta(t)

    # Main sum — the tensor
    Z = 0.0
    for n in range(1, N + 1):
        Z += np.cos(theta - t * np.log(n)) / np.sqrt(n)
    Z *= 2

    # Riemann-Siegel remainder (C₀ correction)
    p = np.sqrt(t / TWO_PI) - N
    # C₀(p) ≈ cos(2π(p² - p - 1/16)) / cos(2πp)
    c2pp = np.cos(2 * np.pi * p)
    if abs(c2pp) > 1e-12:
        C0 = np.cos(2 * np.pi * (p * p - p - 1.0/16)) / c2pp
    else:
        C0 = 0.0
    Z += (-1)**(N - 1) * (t / TWO_PI)**(-0.25) * C0

    return Z


def Z_derivative(t, dt=1e-6):
    """Numerical derivative of Z(t)."""
    return (riemann_siegel_Z(t + dt) - riemann_siegel_Z(t - dt)) / (2 * dt)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: PROCESSOR — Ramanujan refinement
# ═══════════════════════════════════════════════════════════════════════

def ramanujan_refine(n, t0, max_iter=20, tol=1e-12):
    """Newton iteration on N_smooth(T) = n.

    Given a Lambert W estimate t0, iterate:
        t_{k+1} = t_k + (n - N_smooth(t_k)) / N'_smooth(t_k)

    Converges to the smooth coordinate where the counting function
    equals exactly n. Typically 3-5 iterations to machine precision.

    Returns (t_refined, iterations).
    """
    t = t0
    for i in range(max_iter):
        N = smooth_count(t)
        dN = smooth_count_derivative(t)
        if abs(dN) < 1e-20:
            break
        dt = (n - N) / dN
        t += dt
        if abs(dt) < tol:
            return t, i + 1
    return t, max_iter


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: TARGETER — Z(t) search + sequential indexing + Newton
# ═══════════════════════════════════════════════════════════════════════

def _bisect_zero(t_left, t_right, tol=1e-13):
    """Bisect a bracketed sign change of Z(t) to find the zero."""
    for _ in range(70):
        t_mid = (t_left + t_right) / 2
        if riemann_siegel_Z(t_mid) * riemann_siegel_Z(t_left) < 0:
            t_right = t_mid
        else:
            t_left = t_mid
        if t_right - t_left < tol:
            break
    return (t_left + t_right) / 2


def _newton_snap(t, max_iter=15, tol=1e-12):
    """Newton's method to snap to exact zero of Z(t)."""
    for i in range(max_iter):
        Z = riemann_siegel_Z(t)
        if abs(Z) < 1e-14:
            return t, i + 1
        Zp = Z_derivative(t)
        if abs(Zp) < 1e-20:
            break
        dt = Z / Zp
        max_step = 1.0
        if abs(dt) > max_step:
            dt = max_step * np.sign(dt)
        t -= dt
        if abs(dt) < tol:
            return t, i + 1
    return t, max_iter


def find_zero_by_index(n, t_est):
    """Find the nth zero using sequential indexing.

    Strategy:
      1. Search a wide window (±3.5 spacings) around t_est
      2. Find ALL zeros (sign changes of Z)
      3. Sort by position — ordering is exact
      4. Compute base index from median N_smooth + 0.5 bias
      5. Select the candidate at position (n - base)

    The +0.5 bias corrects for S(t) > 0 on average: actual zeros
    sit below their smooth coordinate, so N_smooth(t_n) ≈ n - S
    where S ∈ (0, 1). Adding 0.5 centers the rounding.

    Returns (t_zero, newton_iterations).
    """
    # Local spacing
    if t_est > 10:
        local_spacing = TWO_PI / np.log(t_est / TWO_PI)
    else:
        local_spacing = 8.0
    search_radius = local_spacing * 3.5

    # Grid step: spacing / 4 (catches every zero)
    step = local_spacing / 4
    n_steps = max(30, int(2 * search_radius / step))
    n_steps = min(n_steps, 500)

    t_lo = max(0.5, t_est - search_radius)
    t_hi = t_est + search_radius
    ts = np.linspace(t_lo, t_hi, n_steps)
    Zs = np.array([riemann_siegel_Z(t) for t in ts])

    # Find all sign changes → bracket each zero
    candidates = []
    for i in range(len(Zs) - 1):
        if Zs[i] * Zs[i + 1] < 0:
            t_zero = _bisect_zero(ts[i], ts[i + 1])
            ns = smooth_count(t_zero)
            candidates.append((t_zero, ns))

    if not candidates:
        t_zero, nit = _newton_snap(t_est)
        return t_zero, nit

    # Sequential indexing: sort by position, compute base from N_smooth.
    candidates.sort(key=lambda c: c[0])

    # base + i ≈ true index of candidate i.
    # N_smooth(z_i) ≈ (base + i) - S_i, so base ≈ N_smooth(z_i) - i + 0.5
    base_estimates = [c[1] - i + 0.5 for i, c in enumerate(candidates)]
    base = int(round(np.median(base_estimates)))

    target_idx = n - base
    if 0 <= target_idx < len(candidates):
        t_zero = candidates[target_idx][0]
    else:
        # Fallback: pick closest N_smooth to n
        best = min(candidates, key=lambda c: abs(c[1] - n))
        t_zero = best[0]

    # Newton polish
    t_zero, nit = _newton_snap(t_zero)
    return t_zero, nit


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def zetazero(n, high_precision=False, dps=50):
    """Compute the nth Riemann zeta zero (imaginary part).

    Uses the geometric pipeline:
      Lambert W → Ramanujan refinement → Z(t) + sequential indexing

    Args:
        n: Zero index (1-indexed, positive integer)
        high_precision: If True, polish with mpmath (requires mpmath)
        dps: Decimal places for high-precision mode

    Returns:
        Imaginary part of the nth zeta zero (float, or mpf if high_precision)
    """
    # Stage 1: Lambert W — O(1) estimate
    t_lambert = lambert_w_estimate(n)

    # Stage 2: Ramanujan refinement — exact smooth coordinate
    t_smooth, _ = ramanujan_refine(n, t_lambert)

    # Stage 3: Z(t) + sequential indexing + Newton
    t_zero, _ = find_zero_by_index(n, t_smooth)

    if high_precision:
        return _mpmath_polish(t_zero, dps=dps)

    return t_zero


def zetazero_batch(start, end, high_precision=False, dps=50):
    """Compute a batch of zeta zeros.

    Args:
        start: Starting index (inclusive, 1-indexed)
        end: Ending index (inclusive)
        high_precision: If True, polish each zero with mpmath
        dps: Decimal places for high-precision mode

    Returns:
        Dictionary {n: zero_value}
    """
    results = {}
    for n in range(start, end + 1):
        results[n] = zetazero(n, high_precision=high_precision, dps=dps)
    return results


def _mpmath_polish(t_guess, dps=50):
    """Polish a zero using mpmath's high-precision zeta evaluation.

    This takes our geometric estimate (~1e-4 precision) and refines
    it to arbitrary precision using Newton iteration with mpmath.
    """
    from mpmath import zeta as mp_zeta, mp, mpf, mpc, im, fabs

    original_dps = mp.dps
    mp.dps = dps

    try:
        t = mpf(t_guess)
        tol = mpf(10) ** (-(dps - 5))

        # Newton with cached ζ' (same optimization as fast_zetas.py)
        s_init = mpc('0.5', t)
        zp_cached = mp_zeta(s_init, derivative=1)

        for i in range(10):
            s = mpc('0.5', t)
            z = mp_zeta(s)
            correction = z / zp_cached
            t_new = t - im(correction)

            if i == 0 and dps > 25:
                # Recompute derivative at higher precision
                s_init = mpc('0.5', t_new)
                zp_cached = mp_zeta(s_init, derivative=1)

            if fabs(t_new - t) < tol:
                break
            t = t_new

        return t
    finally:
        mp.dps = original_dps


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def diagnose(n):
    """Show the full pipeline diagnostics for zero #n.

    Returns a dict with all intermediate values.
    """
    t0 = time.time()

    t_lambert = lambert_w_estimate(n)
    t_smooth, refine_iter = ramanujan_refine(n, t_lambert)
    t_zero, snap_iter = find_zero_by_index(n, t_smooth)

    elapsed = time.time() - t0

    N_terms = int(np.sqrt(t_zero / TWO_PI))
    spacing = TWO_PI / np.log(t_zero / TWO_PI) if t_zero > TWO_PI else float('inf')

    return {
        'n': n,
        't_lambert': t_lambert,
        't_smooth': t_smooth,
        't_zero': t_zero,
        'Z_at_zero': riemann_siegel_Z(t_zero),
        'N_smooth_at_zero': smooth_count(t_zero),
        'N_terms': N_terms,
        'local_spacing': spacing,
        'refine_iterations': refine_iter,
        'snap_iterations': snap_iter,
        'time_ms': elapsed * 1000,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def benchmark():
    """Benchmark the geometric hunter."""
    print("=" * 72)
    print("GEOMETRIC ZETA ZERO HUNTER — Benchmark")
    print("=" * 72)
    print()

    # ── Verify against known zeros ──
    print("Test 1: First 30 known zeros")
    print("-" * 72)

    correct = 0
    max_err = 0.0
    t0 = time.time()
    for i, known in enumerate(KNOWN_ZEROS):
        n = i + 1
        found = zetazero(n)
        err = abs(found - known)
        if err < 0.5:
            correct += 1
        max_err = max(max_err, err)
    elapsed = time.time() - t0

    print(f"  Correct index: {correct}/30")
    print(f"  Max |error|:   {max_err:.2e}")
    print(f"  Time:          {elapsed:.3f}s ({elapsed/30*1000:.1f}ms per zero)")
    print()

    # ── Batch test ──
    print("Test 2: Zeros 1-100")
    print("-" * 72)

    t0 = time.time()
    zeros = zetazero_batch(1, 100)
    elapsed = time.time() - t0

    # Check monotonicity
    vals = [zeros[n] for n in range(1, 101)]
    diffs = [vals[i+1] - vals[i] for i in range(99)]
    monotone = all(d > 0 for d in diffs)

    print(f"  Zeros found:   100")
    print(f"  Monotone:      {monotone}")
    print(f"  Min spacing:   {min(diffs):.6f}")
    print(f"  Time:          {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per zero)")
    print()

    # ── High zeros ──
    print("Test 3: High zeros (direct by index)")
    print("-" * 72)

    for n in [200, 500, 1000, 5000, 10000]:
        t0 = time.time()
        t = zetazero(n)
        elapsed = time.time() - t0
        ns = smooth_count(t)
        Z = riemann_siegel_Z(t)
        N_terms = int(np.sqrt(t / TWO_PI))
        print(f"  n={n:6d}: t={t:16.8f}, |Z|={abs(Z):.1e}, "
              f"N_smooth={ns:.2f}, N_terms={N_terms}, "
              f"time={elapsed*1000:.1f}ms")
    print()

    # ── Expanding tensor ──
    print("The Expanding Tensor:")
    print("-" * 72)
    print(f"  {'t':>10s}  {'N_terms':>8s}  {'spacing':>10s}  {'density':>10s}")
    for t in [14, 50, 100, 500, 1000, 5000, 10000]:
        N = int(np.sqrt(t / TWO_PI))
        spacing = TWO_PI / np.log(t / TWO_PI) if t > TWO_PI else float('inf')
        density = 1.0 / spacing if spacing < float('inf') else 0.0
        print(f"  {t:10.1f}  {N:8d}  {spacing:10.4f}  {density:10.4f}")
    print()
    print("  Each term = a rotation axis in the tensor.")
    print("  N_terms grows as sqrt(t/2pi). The tensor expands with time.")

    # ── mpmath comparison (if available) ──
    try:
        import mpmath
        print()
        print("Test 4: Verify against mpmath.zetazero")
        print("-" * 72)
        mpmath.mp.dps = 20
        correct = 0
        errs = []
        for n in range(1, 101):
            mp_zero = float(mpmath.zetazero(n).imag)
            ours = zeros[n]
            err = abs(ours - mp_zero)
            errs.append(err)
            if err < 0.5:
                correct += 1
        print(f"  Index-accurate: {correct}/100")
        print(f"  Mean |error|:   {np.mean(errs):.2e}")
        print(f"  Max |error|:    {np.max(errs):.2e}")
        print(f"  < 1e-3:         {sum(1 for e in errs if e < 1e-3)}/100")
        print(f"  < 1e-4:         {sum(1 for e in errs if e < 1e-4)}/100")
    except ImportError:
        print()
        print("  (mpmath not available — skipping verification)")

    print()
    print("=" * 72)


def compare_with_original():
    """Compare geometric hunter with the original fast_zetas.py."""
    try:
        from fast_zetas import zetazero as original_zetazero
    except ImportError:
        print("Cannot import fast_zetas.py — run from the rhzeros directory.")
        return

    print("=" * 72)
    print("COMPARISON: Geometric vs Original (fast_zetas.py)")
    print("=" * 72)
    print()

    print(f"  {'n':>6s}  {'Original':>18s}  {'Geometric':>18s}  {'Diff':>12s}")
    print(f"  {'─'*6}  {'─'*18}  {'─'*18}  {'─'*12}")

    for n in [1, 10, 50, 100, 200, 500]:
        t0 = time.time()
        geo = zetazero(n)
        t_geo = time.time() - t0

        t0 = time.time()
        orig = float(original_zetazero(n))
        t_orig = time.time() - t0

        diff = geo - orig
        print(f"  {n:6d}  {orig:18.10f}  {geo:18.10f}  {diff:+12.2e}")

    print()
    print("  Original: Lambert W + empirical harmonics + mpmath Newton (50 dps)")
    print("  Geometric: Lambert W + exact θ Ramanujan + R-S Z(t) sequential")
    print()
    print("  Original gives ~50 digit precision (uses mpmath ζ evaluation).")
    print("  Geometric gives ~1e-4 precision (uses Riemann-Siegel formula).")
    print("  Geometric + high_precision mode combines both for best of both worlds.")
    print()
    print("=" * 72)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Geometric Zeta Zero Hunter — find zeros by index using pure geometry')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark suite')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with original fast_zetas.py')
    parser.add_argument('--n', type=int, default=None,
                       help='Find the nth zero')
    parser.add_argument('--batch', type=str, default=None,
                       help='Batch range (e.g., "1-100")')
    parser.add_argument('--diagnose', type=int, default=None,
                       help='Show full pipeline diagnostics for zero #n')
    parser.add_argument('--high-precision', action='store_true',
                       help='Use mpmath polish for high precision')
    parser.add_argument('--dps', type=int, default=50,
                       help='Decimal places for high-precision mode')

    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    elif args.compare:
        compare_with_original()
    elif args.diagnose is not None:
        d = diagnose(args.diagnose)
        print(f"Zero #{d['n']}:")
        print(f"  Lambert W estimate: {d['t_lambert']:.10f}")
        print(f"  Smooth coordinate:  {d['t_smooth']:.10f}  ({d['refine_iterations']} iterations)")
        print(f"  Found zero:         {d['t_zero']:.10f}  ({d['snap_iterations']} Newton steps)")
        print(f"  |Z(t)|:            {abs(d['Z_at_zero']):.2e}")
        print(f"  N_smooth at zero:   {d['N_smooth_at_zero']:.4f}")
        print(f"  R-S terms (tensor): {d['N_terms']}")
        print(f"  Local spacing:      {d['local_spacing']:.4f}")
        print(f"  Time:               {d['time_ms']:.1f}ms")
    elif args.batch:
        start, end = map(int, args.batch.split('-'))
        print(f"Computing zeros {start} to {end}...")
        t0 = time.time()
        zeros = zetazero_batch(start, end,
                              high_precision=args.high_precision,
                              dps=args.dps)
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.3f}s ({elapsed/(end-start+1)*1000:.1f}ms per zero)")
        print()
        for n in sorted(zeros.keys())[:20]:
            print(f"  zetazero({n}) = {zeros[n]}")
        if len(zeros) > 20:
            print(f"  ... ({len(zeros) - 20} more)")
    elif args.n is not None:
        t = zetazero(args.n, high_precision=args.high_precision, dps=args.dps)
        print(f"zetazero({args.n}) = {t}")
    else:
        parser.print_help()
