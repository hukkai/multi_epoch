# Cold-Start Block-Dual Corrected Global Muon

This README describes the first implementation pass for **block-dual corrected global Muon**.

The goal is to keep the optimizer change minimal:

- Add exactly one user-facing option:

```python
dual_init: Literal["none", "zero", "first_order"] = "none"
```

- Do **not** add warm-start state.
- Do **not** expose `block_dual_lr`, `block_dual_steps`, or other new hyperparameters.
- Fix all internal constants to recommended validation values.
- Preserve the current baseline behavior when `dual_init="none"`.

The purpose of this first pass is to answer one question:

> Does solving a more accurate block-tangent, global-Muon direction improve over the current baseline?

If this cold-start, many-step version does not improve, warm-start versions are unlikely to be worth adding immediately.

---

## 1. High-level behavior

Assume the optimizer already has a baseline path like:

```python
P = global_msign(M)
D = block_tangent_project(W, P)
D = optional_global_spectral_cap(D)
W = blockwise_retraction(W - lr * D)
```

where:

- `M` is the exact tensor currently passed into the global Muon polar / matrix sign operation.
- `global_msign` is the existing full-matrix Muon orthogonalization / matrix-sign routine.
- `block_tangent_project` projects each constrained block onto its Stiefel tangent space.
- `blockwise_retraction` reorthogonalizes each constrained block.

The new behavior is:

```python
if dual_init == "none":
    # exact current baseline
    P = global_msign(M)
else:
    # cold-start block-dual correction
    P = block_dual_corrected_global_msign(W_blocks, M_blocks, dual_init)

D = block_tangent_project(W_blocks, P_blocks)
D = global_spectral_cap(D)
W = blockwise_retraction(W - lr * D)
```

Important:

```python
# Always global, never blockwise:
P = global_msign(pack(Z_blocks))
```

Do **not** replace this with:

```python
P_b = global_msign(Z_b)  # wrong for this method
```

The whole point is:

> Muon polar / matrix-sign is global; Stiefel constraints and retractions are blockwise.

---

## 2. User-facing option

Add one optimizer option:

```python
dual_init: str = "none"
```

Valid values:

```python
"none"
"zero"
"first_order"
```

Recommended default:

```python
dual_init = "none"
```

This preserves existing behavior by default.

### `dual_init="none"`

No dual correction. This is the current baseline.

```python
P = global_msign(M)
D_b = Proj_Tb(P_b)
```

Use this to verify the refactor does not change current training.

### `dual_init="zero"`

Cold-start dual correction with:

```python
Lambda_b = 0
```

for every block, at every optimizer step.

No history is saved.

### `dual_init="first_order"`

Cold-start dual correction with a first-order analytic initialization for every block, at every optimizer step.

No history is saved.

For column-Stiefel blocks:

```python
Lambda_b = -0.25 * sym(W_b.T @ M_b + M_b.T @ W_b)
```

For row-Stiefel blocks:

```python
Lambda_b = -0.25 * sym(M_b @ W_b.T + W_b @ M_b.T)
```

---

## 3. Fixed internal constants

Do not expose these in the optimizer constructor for this first pass.

Use:

```python
BLOCK_DUAL_STEPS = 20
BLOCK_DUAL_LR = 0.1
FINAL_BLOCK_PROJECTION = True
GLOBAL_SPECTRAL_CAP_AFTER_PROJECTION = True
SPECTRAL_CAP_POWER_ITERS = 2
DUAL_COMPUTE_DTYPE = torch.float32
```

Rationale:

- `BLOCK_DUAL_STEPS = 20` is intentionally large for the validation pass.
- `BLOCK_DUAL_LR = 0.1` is conservative enough for the nonlinear `global_msign` loop.
- Final block tangent projection should remain enabled as a safety net.
- Global spectral cap should remain enabled after projection so dual correction does not change the effective trust-region scale.
- Dual correction should be computed in FP32 even when model weights are lower precision.

Do not add:

```python
block_dual_lambdas
warm_start
warm_init
dual_momentum
```

or any persistent dual state.

---

## 4. Mathematical objective

Let the full matrix be partitioned into constrained blocks:

```text
W = pack(W_1, ..., W_B)
M = pack(M_1, ..., M_B)
```

Each block is constrained on a Stiefel manifold.

For column-Stiefel blocks:

```text
W_b^T W_b = I
```

The tangent condition is:

```text
W_b^T D_b + D_b^T W_b = 0
```

For row-Stiefel blocks:

```text
W_b W_b^T = I
```

The tangent condition is:

```text
D_b W_b^T + W_b D_b^T = 0
```

The desired direction is approximately:

```text
maximize    <M, D>
subject to  D is in every block tangent space
            ||D||_2 <= 1 globally
```

This method approximates that direction by adding blockwise normal-space dual corrections before a global Muon matrix sign:

Column-Stiefel:

```text
Z_b = M_b + 2 W_b Lambda_b
P   = global_msign(pack(Z_b))
```

Row-Stiefel:

```text
Z_b = M_b + 2 Lambda_b W_b
P   = global_msign(pack(Z_b))
```

Then each `Lambda_b` is updated to reduce the tangent residual of the resulting `P_b`.

---

## 5. Column-Stiefel implementation

Use this when each block satisfies:

```python
W_b.T @ W_b ~= I
```

### Initialization

```python
def sym(A):
    return 0.5 * (A + A.transpose(-1, -2))

if dual_init == "zero":
    Lambda_b = torch.zeros(
        W_b.shape[1], W_b.shape[1],
        device=W_b.device,
        dtype=torch.float32,
    )
elif dual_init == "first_order":
    W32 = W_b.float()
    M32 = M_b.float()
    Lambda_b = -0.25 * sym(W32.transpose(-1, -2) @ M32 + M32.transpose(-1, -2) @ W32)
else:
    raise ValueError(f"unknown dual_init: {dual_init}")
```

### Inner loop

```python
lambdas = init_lambdas(W_blocks, M_blocks, dual_init, orientation="column")

for _ in range(BLOCK_DUAL_STEPS):
    Z_blocks = []

    for W_b, M_b, Lambda_b in zip(W_blocks, M_blocks, lambdas):
        W32 = W_b.float()
        M32 = M_b.float()
        Z_b = M32 + 2.0 * (W32 @ Lambda_b)
        Z_blocks.append(Z_b.to(dtype=M_b.dtype))

    Z = pack_blocks_like_original_matrix(Z_blocks)
    P = global_msign(Z)
    P_blocks = unpack_blocks_like_original_matrix(P)

    new_lambdas = []

    for W_b, P_b, Lambda_b in zip(W_blocks, P_blocks, lambdas):
        W32 = W_b.float()
        P32 = P_b.float()

        H_b = W32.transpose(-1, -2) @ P32 + P32.transpose(-1, -2) @ W32
        H_b = sym(H_b)

        Lambda_b = Lambda_b - BLOCK_DUAL_LR * H_b
        Lambda_b = sym(Lambda_b)

        new_lambdas.append(Lambda_b)

    lambdas = new_lambdas
```

### Final global msign after the last Lambda update

The inner loop updates `Lambda_b` at the end of each iteration. Therefore, after the loop, recompute `P` once using the final `Lambda_b` values:

```python
Z_blocks = []

for W_b, M_b, Lambda_b in zip(W_blocks, M_blocks, lambdas):
    W32 = W_b.float()
    M32 = M_b.float()
    Z_b = M32 + 2.0 * (W32 @ Lambda_b)
    Z_blocks.append(Z_b.to(dtype=M_b.dtype))

Z = pack_blocks_like_original_matrix(Z_blocks)
P = global_msign(Z)
P_blocks = unpack_blocks_like_original_matrix(P)
```

### Final tangent projection

Keep the final projection even after dual correction:

```python
D_blocks = []

for W_b, P_b in zip(W_blocks, P_blocks):
    W32 = W_b.float()
    P32 = P_b.float()
    D_b = P32 - W32 @ sym(W32.transpose(-1, -2) @ P32)
    D_blocks.append(D_b.to(dtype=P_b.dtype))
```

---

## 6. Row-Stiefel implementation

Use this when each block satisfies:

```python
W_b @ W_b.T ~= I
```

### Initialization

```python
if dual_init == "zero":
    Lambda_b = torch.zeros(
        W_b.shape[0], W_b.shape[0],
        device=W_b.device,
        dtype=torch.float32,
    )
elif dual_init == "first_order":
    W32 = W_b.float()
    M32 = M_b.float()
    Lambda_b = -0.25 * sym(M32 @ W32.transpose(-1, -2) + W32 @ M32.transpose(-1, -2))
else:
    raise ValueError(f"unknown dual_init: {dual_init}")
```

### Inner loop

```python
lambdas = init_lambdas(W_blocks, M_blocks, dual_init, orientation="row")

for _ in range(BLOCK_DUAL_STEPS):
    Z_blocks = []

    for W_b, M_b, Lambda_b in zip(W_blocks, M_blocks, lambdas):
        W32 = W_b.float()
        M32 = M_b.float()
        Z_b = M32 + 2.0 * (Lambda_b @ W32)
        Z_blocks.append(Z_b.to(dtype=M_b.dtype))

    Z = pack_blocks_like_original_matrix(Z_blocks)
    P = global_msign(Z)
    P_blocks = unpack_blocks_like_original_matrix(P)

    new_lambdas = []

    for W_b, P_b, Lambda_b in zip(W_blocks, P_blocks, lambdas):
        W32 = W_b.float()
        P32 = P_b.float()

        H_b = P32 @ W32.transpose(-1, -2) + W32 @ P32.transpose(-1, -2)
        H_b = sym(H_b)

        Lambda_b = Lambda_b - BLOCK_DUAL_LR * H_b
        Lambda_b = sym(Lambda_b)

        new_lambdas.append(Lambda_b)

    lambdas = new_lambdas
```

### Final global msign after the last Lambda update

```python
Z_blocks = []

for W_b, M_b, Lambda_b in zip(W_blocks, M_blocks, lambdas):
    W32 = W_b.float()
    M32 = M_b.float()
    Z_b = M32 + 2.0 * (Lambda_b @ W32)
    Z_blocks.append(Z_b.to(dtype=M_b.dtype))

Z = pack_blocks_like_original_matrix(Z_blocks)
P = global_msign(Z)
P_blocks = unpack_blocks_like_original_matrix(P)
```

### Final tangent projection

```python
D_blocks = []

for W_b, P_b in zip(W_blocks, P_blocks):
    W32 = W_b.float()
    P32 = P_b.float()
    D_b = P32 - sym(P32 @ W32.transpose(-1, -2)) @ W32
    D_blocks.append(D_b.to(dtype=P_b.dtype))
```

---

## 7. Sign convention

Use the exact same `M` that the current baseline sends into `global_msign`.

For example, if the baseline is:

```python
P = global_msign(M)
W = W - lr * project(P)
```

then dual correction must use:

```python
M_b
```

with the same sign.

If the baseline is:

```python
P = global_msign(-M)
W = W + lr * project(P)
```

then dual correction must use:

```python
M_input = -M
```

The first-order initialization must always be computed from the actual matrix passed into `global_msign`.

---

## 8. Global spectral cap after projection

After final tangent projection, repack the direction:

```python
D = pack_blocks_like_original_matrix(D_blocks)
```

Then apply a global spectral cap:

```python
sigma = spectral_norm_estimate(D, n_iters=SPECTRAL_CAP_POWER_ITERS)
D = D / torch.clamp(sigma, min=1.0)
```

If the existing code already has an equivalent post-projection cap, reuse it.

Do not apply independent caps per block. The cap should be global on the packed matrix.

---

## 9. Minimal optimizer diff

The intended integration should look approximately like this:

```python
class Optimizer(...):
    def __init__(self, ..., dual_init: str = "none", ...):
        if dual_init not in {"none", "zero", "first_order"}:
            raise ValueError("dual_init must be one of: 'none', 'zero', 'first_order'")
        self.dual_init = dual_init
```

Then at the update site:

```python
M = get_current_muon_input(...)
W_blocks = split_param_into_constraint_blocks(W)
M_blocks = split_tensor_like_param_blocks(M)

if self.dual_init == "none":
    P = global_msign(M)
    P_blocks = split_tensor_like_param_blocks(P)
else:
    P_blocks = cold_start_block_dual_global_msign(
        W_blocks=W_blocks,
        M_blocks=M_blocks,
        dual_init=self.dual_init,
        orientation=stiefel_orientation,
        global_msign=global_msign,
        pack_blocks=pack_blocks_like_original_matrix,
        unpack_blocks=unpack_blocks_like_original_matrix,
    )

D_blocks = block_tangent_project(W_blocks, P_blocks, orientation=stiefel_orientation)
D = pack_blocks_like_original_matrix(D_blocks)
D = global_spectral_cap(D)
D_blocks = unpack_blocks_like_original_matrix(D)
W = blockwise_retraction(W, D_blocks, lr=lr, orientation=stiefel_orientation)
```

Do not change the baseline path when `dual_init="none"`.

---

## 10. Diagnostics to add

Add these diagnostics if there is already a logging/debug mechanism. They should not be required for normal training.

### Pre-projection tangent residual

Column-Stiefel:

```python
H_b = W_b.T @ P_b + P_b.T @ W_b
r_b = H_b.norm() / math.sqrt(H_b.shape[0])
```

Row-Stiefel:

```python
H_b = P_b @ W_b.T + W_b @ P_b.T
r_b = H_b.norm() / math.sqrt(H_b.shape[0])
```

Log:

```python
block_dual_pre_tangent_residual_max = max_b r_b
block_dual_pre_tangent_residual_mean = mean_b r_b
```

Expected behavior:

- `dual_init="zero"` or `"first_order"` should usually reduce this residual as `BLOCK_DUAL_STEPS` increases.
- If it grows or becomes NaN, first check the sign of the Lambda update and the `M` sign convention.

### Inner objective

Compare baseline projected direction and dual-corrected projected direction:

```python
obj = torch.sum(M * D)
```

Log:

```python
block_dual_inner_obj = torch.sum(M * D_dual)
block_dual_inner_obj_baseline = torch.sum(M * D_baseline)
block_dual_inner_obj_ratio = obj_dual / (obj_baseline + eps)
```

Expected behavior:

- The ratio should not be systematically below `1.0`.
- If residual decreases but objective gets worse, the dual correction may be over-solving in a direction that hurts descent alignment.

### Direction difference from baseline

```python
direction_delta = (D_dual - D_baseline).norm() / (D_baseline.norm() + eps)
```

Expected behavior:

- If this is near zero, dual correction cannot change training much.
- If this is very large and training destabilizes, try temporarily reducing `BLOCK_DUAL_LR` internally from `0.1` to `0.05`, but do not expose it as a public option in this first pass.

---

## 11. Unit tests

Add small tests before running expensive training.

### Test 1: `dual_init="none"` preserves baseline

With a fixed seed and fixed input:

```python
D_old = old_baseline_direction(W, M)
D_new = new_direction(W, M, dual_init="none")
assert close(D_old, D_new)
```

This should be exact or nearly exact, depending on floating-point refactor details.

### Test 2: Lambda shapes

Column-Stiefel block:

```python
W_b.shape == (m, n)
Lambda_b.shape == (n, n)
```

Row-Stiefel block:

```python
W_b.shape == (m, n)
Lambda_b.shape == (m, m)
```

### Test 3: first-order initialization reduces raw linearized normal residual

Column-Stiefel:

```python
Z_b = M_b + 2 * W_b @ Lambda_b
H_raw = W_b.T @ M_b + M_b.T @ W_b
H_z = W_b.T @ Z_b + Z_b.T @ W_b
assert norm(H_z) < norm(H_raw)
```

For perfectly orthonormal `W_b`, this should nearly cancel the linearized residual.

Row-Stiefel uses:

```python
Z_b = M_b + 2 * Lambda_b @ W_b
H_raw = M_b @ W_b.T + W_b @ M_b.T
H_z = Z_b @ W_b.T + W_b @ Z_b.T
assert norm(H_z) < norm(H_raw)
```

### Test 4: final projected direction is tangent

Column-Stiefel:

```python
D_b = P_b - W_b @ sym(W_b.T @ P_b)
H_b = W_b.T @ D_b + D_b.T @ W_b
assert norm(H_b) < tolerance
```

Row-Stiefel:

```python
D_b = P_b - sym(P_b @ W_b.T) @ W_b
H_b = D_b @ W_b.T + W_b @ D_b.T
assert norm(H_b) < tolerance
```

---

## 12. Failure modes and first checks

### Residual does not decrease

Check:

1. Is `P = global_msign(pack(Z_blocks))` global, not blockwise?
2. Is `M_b` the same sign as the existing baseline `global_msign` input?
3. Is the Lambda update sign correct?

The intended update is:

```python
Lambda_b = Lambda_b - 0.1 * H_b
```

4. Is the Stiefel orientation correct?

Column and row formulas are different.

### Training becomes unstable

Check:

1. Is final tangent projection still enabled?
2. Is global spectral cap applied after projection?
3. Is the cap global over the packed large matrix, not blockwise?
4. Is the dual correction computed in FP32?

### No training difference from baseline

Check:

1. Is `direction_delta = ||D_dual - D_baseline|| / ||D_baseline||` near zero?
2. Is the block tangent residual already tiny before dual correction?
3. Are the constrained blocks too small or too close to already-tangent updates?

### Objective worse than baseline

If:

```python
inner_obj_dual < inner_obj_baseline
```

systematically, then the implementation is not improving the intended inner problem. First verify sign convention and spectral cap. If those are correct, this method may not be useful for the current block layout.

---

## 13. Expected ablation

Run three variants:

```python
dual_init="none"         # baseline
dual_init="zero"         # cold-start zero dual solve
dual_init="first_order"  # cold-start first-order dual solve
```

Use the same fixed constants for the two dual variants:

```python
BLOCK_DUAL_STEPS = 20
BLOCK_DUAL_LR = 0.1
FINAL_BLOCK_PROJECTION = True
GLOBAL_SPECTRAL_CAP_AFTER_PROJECTION = True
```

Interpretation:

- If `first_order` improves over `none`, the block-dual correction is promising.
- If `zero` improves but `first_order` does not, inspect first-order sign and orientation.
- If neither `zero` nor `first_order` improves despite lower tangent residual, the more accurate block-dual direction may not be beneficial for this model or block layout.
- If dual variants improve but are too slow, add warm-start and reduce steps in a later implementation pass.

---

## 14. Non-goals for this pass

Do not implement these yet:

- persistent `Lambda_b` optimizer state;
- warm-start dual variables;
- user-facing `block_dual_lr`;
- user-facing `block_dual_steps`;
- adaptive dual solver tolerance;
- ADMM solver;
- blockwise Muon;
- per-block spectral caps.

This pass should be a clean, cold-start validation of:

```text
global polar + block-dual correction + final block projection + block retraction
```
