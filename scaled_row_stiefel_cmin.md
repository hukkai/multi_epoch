# Implement scaled row-Stiefel constraint: `XX^T = c I`, `c >= 0.5`

## Goal

For each constrained matrix block `X` with shape `(..., n, m)` and `n < m`, replace the current fixed Stiefel constraint

```text
X @ X.T = I_n
```

with the relaxed scaled row-Stiefel constraint

```text
X @ X.T = c * I_n, where c >= c_min and c_min = 0.5
```

Here `c` is **not fixed**. It is allowed to change during training, but all rows must remain mutually orthogonal with the same row norm.

---

## Current scale

For a valid `X`, compute

```text
c = trace(X @ X.T) / n = ||X||_F^2 / n
```

In code, for batched tensors:

```python
c = X.square().sum(dim=(-2, -1), keepdim=True) / X.shape[-2]
```

Use `c.clamp_min(c_min)` when needed for numerical safety.

---

## Tangent-space projection

Given a raw optimizer update `Delta` from Adam/Muon/etc., project it to the tangent space of

```text
{X : X @ X.T = c I_n, c free}
```

The tangent condition is

```text
sym(U @ X.T) = mu * I_n
```

where `mu` is a scalar and

```text
sym(A) = 0.5 * (A + A.T)
```

Projection formula:

```text
S  = sym(Delta @ X.T)
mu = trace(S) / n
U  = Delta - (1 / c) * (S - mu * I_n) @ X
```

Important: this differs from fixed Stiefel `XX^T = I`, where the whole `S @ X` term is removed. Here only the traceless symmetric part is removed, so the global scaling direction is preserved.

### Boundary handling for `c >= 0.5`

Normally it is enough to enforce `c >= 0.5` in the retraction step. If handling the boundary in projection, then when `c` is already near `c_min`, prevent directions that decrease `c`:

```text
if c <= c_min + tol:
    mu = max(mu, 0)
```

Then use the same projection formula with the clamped `mu`.

Recommended default: enforce the lower bound in retraction, not in projection.

---

## Retraction

After projection:

```text
W = X + U
```

Retract `W` back to the constraint set.

First compute row-polar normalization:

```text
Q = (W @ W.T)^(-1/2) @ W
```

so that

```text
Q @ Q.T = I_n
```

Then choose the new scale:

```text
c_new = max(||W||_F^2 / n, c_min)
```

Finally:

```text
X_new = sqrt(c_new) * Q
```

This guarantees:

```text
X_new @ X_new.T = c_new * I_n
c_new >= 0.5
```

---

## Implementation notes

Use the existing polar/eigh machinery if available.

For row-polar with `W` shape `(..., n, m)`:

```python
G = W @ W.transpose(-1, -2)                    # (..., n, n)
evals, evecs = torch.linalg.eigh(G.float())
evals = evals.clamp_min(eps)
G_inv_sqrt = (evecs * evals.rsqrt().unsqueeze(-2)) @ evecs.transpose(-1, -2)
Q = G_inv_sqrt.to(W.dtype) @ W
```

Then:

```python
c_new = W.square().sum(dim=(-2, -1), keepdim=True) / W.shape[-2]
c_new = c_new.clamp_min(0.5)
X_new = c_new.sqrt() * Q
```

Need to preserve dtype behavior. Prefer doing the eigendecomposition in `float32`, then cast back to the original dtype for the final matmul/update.

---

## Suggested helper functions

Add or modify helpers roughly as follows:

```python
def project_scaled_row_stiefel_update(
    X: torch.Tensor,
    Delta: torch.Tensor,
    c_min: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project raw update Delta to tangent space of XX^T = cI, with c free."""
```

```python
def retract_scaled_row_stiefel(
    W: torch.Tensor,
    c_min: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Retract W to {X: XX^T = cI, c >= c_min}."""
```

If there is already a fixed-Stiefel projection/retraction path, keep it and add a separate mode, e.g.

```python
constraint = "stiefel"              # XX^T = I
constraint = "scaled_row_stiefel"   # XX^T = cI, c >= 0.5
```

---

## Validation checks

For each constrained matrix block after retraction, verify:

```python
G = X_new @ X_new.transpose(-1, -2)
c = G.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
I = torch.eye(n, device=X.device, dtype=X.dtype)
err = (G - c[..., None, None] * I).norm() / G.norm().clamp_min(1e-12)
assert c.min() >= 0.5 - tolerance
```

Expected behavior:

1. Rows are mutually orthogonal.
2. All row norms are equal.
3. The shared squared row norm `c` is allowed to change.
4. `c` never drops below `0.5` after retraction.
5. The raw optimizer update can still come from Adam, AdamW, Muon, etc.; only the post-processing changes.
