# rhzeros

Two approaches to finding Riemann zeta zeros by index.

## Files

| File | Approach | Precision | Speed |
|------|----------|-----------|-------|
| `fast_zetas.py` | Lambert W + empirical harmonics + mpmath Newton | ~50 digits | ~1.7ms/zero |
| `geometric_zetas.py` | Lambert W + exact θ Ramanujan + Riemann-Siegel Z(t) | ~1e-4 (float) or ~50 digits (hybrid) | ~5ms/zero |

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Original (fast_zetas.py)

```bash
# Single zero
python fast_zetas.py --n 100

# Batch
python fast_zetas.py --batch 1-100

# Benchmark against mpmath
python fast_zetas.py --benchmark
```

### Geometric (geometric_zetas.py)

```bash
# Single zero
python geometric_zetas.py --n 100

# High precision (geometric estimate + mpmath polish)
python geometric_zetas.py --n 100 --high-precision --dps 50

# Batch
python geometric_zetas.py --batch 1-100

# Full benchmark suite
python geometric_zetas.py --benchmark

# Compare both approaches
python geometric_zetas.py --compare

# Pipeline diagnostics for a single zero
python geometric_zetas.py --diagnose 100
```

### As a library

```python
# Original
from fast_zetas import zetazero, zetazero_batch
z = zetazero(100)  # mpf, ~50 digits

# Geometric
from geometric_zetas import zetazero, zetazero_batch
z = zetazero(100)                              # float, ~1e-4
z = zetazero(100, high_precision=True, dps=50) # mpf, ~50 digits
```

---

## The Two Approaches

### Original: Empirical + mpmath

The original `fast_zetas.py` uses:

1. **Lambert W** base estimate: `T ≈ 2π(n - 11/8) / W((n - 11/8)/e)`
2. **Empirical harmonic corrections**: fitted parameters for 5-fold harmonic structure, logarithmic spiral, and self-interference
3. **mpmath Newton refinement**: evaluates `ζ(1/2 + it)` at 50-digit precision with cached `ζ'` for 3× speedup

The estimate gets within ~0.3 of the true zero, then mpmath Newton snaps to 50 digits. This is fast (1.7ms/zero in batch) because the initial guess is good enough for Newton to converge in 3-5 iterations.

### Geometric: Exact θ + Riemann-Siegel

The geometric `geometric_zetas.py` replaces the empirical corrections and mpmath evaluation with analytic geometry:

1. **Lambert W** base estimate (same as original)
2. **Ramanujan refinement**: Newton iteration on the exact smooth counting function `θ(T)/π + 1 = n`, using `loggamma` for machine-precision θ at all t (the asymptotic Stirling expansion diverges for t < 20)
3. **Riemann-Siegel Z(t)**: evaluates the real-valued zeta function on the critical line as a finite sum of cosines + remainder correction
4. **Sequential indexing**: finds all zeros in a window, sorts by position, assigns indices from median N_smooth + 0.5 bias correction

No Gram points. No sweep of the critical line. Direct: index n → zero t_n.

**100/100 index-accurate** against mpmath, monotone, mean error 1.9×10⁻⁴.

The optional `high_precision=True` mode uses the geometric pipeline to find the right zero, then hands off to mpmath Newton for arbitrary-precision polish — best of both worlds.

---

## The Expanding Tensor

The Riemann-Siegel formula reveals that the zeta function is a **tensor that grows with time**:

```
Z(t) = 2 Σ_{n=1}^{N(t)} n^{-1/2} cos(θ(t) - t·ln(n)) + remainder

where N(t) = floor(√(t/2π))
```

Each term is a rotation. The number of terms (rotation axes) increases with height:

```
     t      terms   spacing   density
    14.0       1     7.84      0.13
   100.0       3     2.27      0.44
  1000.0      12     1.24      0.81
 10000.0      39     0.85      1.17
```

A zero is where all rotations conspire to cancel. At the first zero (t ≈ 14.13), one rotation lands on a null. At zero #10000, 39 rotations must cancel collectively.

This maps onto the transformer architecture:
- **Terms = tokens** (the sequence being processed)
- **Phases = position encodings** (`t·ln(n)` is the zeta analog of RoPE)
- **Amplitudes = embeddings** (`n^{-1/2}` decay)
- **Zero = output** (where the aggregation produces a result)
- **N(t) = context window** (grows as √t)

The three-stage pipeline mirrors the transformer's layer structure:
- **Compressor** (Lambert W): reads global shape, captures >95%
- **Processor** (Ramanujan): oscillatory corrections on smooth geometry
- **Targeter** (Z(t) + Newton): evaluates the full tensor, finds exact output

---

## Dependencies

- `numpy` — array operations
- `scipy` — Lambert W function, loggamma for exact θ
- `mpmath` — high-precision arithmetic (used by `fast_zetas.py` and optional `high_precision` mode in `geometric_zetas.py`)
