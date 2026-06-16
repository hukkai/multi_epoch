# Block-Dual Corrected Global Muon 实现说明

这份文档用于让 Codex 在现有代码中把当前的

```text
global Muon polar / msign -> blockwise tangent projection -> blockwise retraction
```

升级为更 principled 的

```text
block-dual corrected global Muon -> blockwise tangent projection -> blockwise retraction
```

核心要求：**Muon 的 `polar / msign` 仍然在完整大矩阵上做，不要退回到每个小 block 独立做 Muon。** block 只用于 Stiefel 切空间约束、对偶修正和 retraction。

---

## 1. 背景和目标

当前经验最优做法是：

1. 对完整大矩阵的 momentum / gradient-like tensor 做一次 global Muon polar：
   \[
   P = \operatorname{msign}(M).
   \]
2. 把 `P` 按小矩阵 blocks 切开。
3. 对每个 block 做 Stiefel tangent projection。
4. 对每个 block 做 retraction，保持 block-level orthogonality。

这个方法已经比 blockwise Muon 稳定，但它只是近似。原因是：

\[
\operatorname{Proj}_{T\mathcal M}(\operatorname{msign}(M))
\neq
\arg\max_{D \in T\mathcal M,\ \|D\|_2 \le 1} \langle M, D \rangle.
\]

这里的 \(\mathcal M\) 是 product of small Stiefel manifolds。也就是说，**global msign 和 block tangent projection 不可交换**。

目标是保留 global Muon 的稳定性，同时让 global msign 的输入提前带有 blockwise normal correction，使得到的方向更接近 product-Stiefel tangent space。

---

## 2. 要实现的数学对象

设一个大权重矩阵 \(W\) 被切成 blocks：

\[
W = \operatorname{pack}(W_1, \dots, W_B).
\]

每个 block 满足 Stiefel 约束。对 column-Stiefel block：

\[
W_b^\top W_b = I.
\]

对应切空间条件是：

\[
W_b^\top D_b + D_b^\top W_b = 0.
\]

我们希望求一个方向 \(D\)：

\[
D^\star =
\arg\max_D \langle M, D \rangle
\quad
\text{s.t.}
\quad
\|D\|_2 \le 1,
\quad
W_b^\top D_b + D_b^\top W_b = 0,\ \forall b.
\]

这里的关键是：

- spectral norm 约束 \(\|D\|_2 \le 1\) 是对**完整 packed 大矩阵**的约束；
- Stiefel 切空间约束是对每个 block 的局部约束。

因此这不是 full-Stiefel Muon，也不是 blockwise Muon，而是：

```text
Product-Stiefel parameter constraint + global spectral-norm Muon geometry
```

---

## 3. Block-dual correction 的基本公式

对每个 block 存一个对称对偶变量 \(\Lambda_b\)。

### 3.1 Column-Stiefel block

如果 block 是 column-orthonormal：

\[
W_b^\top W_b = I,
\]

则 normal correction 为：

\[
N_b = 2 W_b \Lambda_b.
\]

构造修正后的 global Muon 输入：

\[
Z = \operatorname{pack}(M_b + 2 W_b\Lambda_b)_{b=1}^B.
\]

然后做一次 global matrix sign / polar：

\[
P = \operatorname{msign}_{\text{global}}(Z).
\]

对每个 block 计算 tangent residual：

\[
H_b = W_b^\top P_b + P_b^\top W_b.
\]

更新对偶变量：

\[
\Lambda_b \leftarrow \Lambda_b - \alpha H_b.
\]

其中 \(\alpha\) 是 dual learning rate。默认可以从 `0.05` 或 `0.1` 开始；如果 residual 稳定下降，可以试 `0.25`。

### 3.2 Row-Stiefel block

如果 block 是 row-orthonormal：

\[
W_b W_b^\top = I,
\]

则 normal correction 为：

\[
N_b = 2 \Lambda_b W_b.
\]

global Muon 输入为：

\[
Z = \operatorname{pack}(M_b + 2\Lambda_b W_b)_{b=1}^B.
\]

residual 为：

\[
H_b = W_b P_b^\top + P_b W_b^\top.
\]

对偶更新同样是：

\[
\Lambda_b \leftarrow \Lambda_b - \alpha H_b.
\]

---

## 4. 推荐实现路线

建议实现两个模式。

### 4.1 模式 A：warm-start 1-call 模式，优先实现

这个模式每个 optimizer step 只额外维护 \(\Lambda_b\)，但仍然只做**一次** global `msign`。它最容易接入现有代码。

流程：

```text
1. 从 optimizer state 读取上一 step 的 Lambda_b。
2. 用 Lambda_b 构造 Z_b = M_b + normal_correction(W_b, Lambda_b)。
3. pack Z_b 得到完整大矩阵 Z。
4. 对 Z 做一次 global msign，得到 P。
5. 计算每个 block 的 tangent residual H_b。
6. 用 H_b 更新 Lambda_b，存回 optimizer state，供下一 step warm start。
7. 对 P_b 做 final safe tangent projection，得到 D_b。
8. 对 packed D 做 global spectral norm cap。
9. 对每个 W_b 做 retraction。
```

这个模式的好处：

- 与当前 global polar baseline 的计算量几乎一致；
- 利用上一 step 的 \(\Lambda_b\) 修正当前 step；
- 即使 \(\Lambda_b\) 不够精确，final tangent projection 仍然保证安全。

### 4.2 模式 B：2-call refinement 模式，作为可选增强

如果模式 A 的 tangent residual 仍然较大，可以在同一个 optimizer step 内做一次内层修正：

```text
1. 用旧 Lambda_b 构造 Z。
2. global msign 得到 P。
3. 计算 H_b。
4. Lambda_b <- Lambda_b - alpha * H_b。
5. 用更新后的 Lambda_b 重新构造 Z。
6. 再做一次 global msign，得到最终 P。
7. final safe tangent projection。
8. global spectral norm cap。
9. blockwise retraction。
```

这个模式更接近真正的 dual-corrected solution，但会增加一次 global `msign`。

建议先实现模式 A，再通过配置打开模式 B。

---

## 5. 需要加入的配置项

建议在 optimizer config 或 parameter group 中加入：

```python
use_block_dual_correction: bool = True
block_dual_mode: str = "warmstart_1call"  # or "refine_2call"
block_dual_lr: float = 0.1
block_dual_init: str = "warm"             # "zero" or "warm"
block_dual_symmetrize: bool = True
block_dual_clip_norm: float | None = None
final_tangent_projection: bool = True
final_global_spectral_cap: bool = True
spectral_cap_value: float = 1.0
spectral_norm_power_iters: int = 1
```

Safe default：

```python
use_block_dual_correction = True
block_dual_mode = "warmstart_1call"
block_dual_lr = 0.1
block_dual_init = "warm"
final_tangent_projection = True
final_global_spectral_cap = True
spectral_norm_power_iters = 1
```

如果训练不稳，优先尝试：

```python
block_dual_lr = 0.05
block_dual_mode = "warmstart_1call"
spectral_cap_value = 1.0
```

如果 tangent residual 下降不明显，再尝试：

```python
block_dual_mode = "refine_2call"
block_dual_lr = 0.1 或 0.25
```

---

## 6. Optimizer state

每个被 block-Stiefel 约束的 parameter 需要新增：

```python
state[p]["block_dual_lambdas"] = list[Tensor]
```

其中第 `b` 个 tensor 是 \(\Lambda_b\)。

### 6.1 Column-Stiefel block 的 Lambda shape

如果：

```python
W_b.shape == (m, n)
W_b.T @ W_b = I_n
```

则：

```python
Lambda_b.shape == (n, n)
```

### 6.2 Row-Stiefel block 的 Lambda shape

如果：

```python
W_b.shape == (r, c)
W_b @ W_b.T = I_r
```

则：

```python
Lambda_b.shape == (r, r)
```

### 6.3 dtype

建议：

```python
Lambda_b.dtype = torch.float32
```

即使参数是 bf16 / fp16，也建议 \(\Lambda_b\)、residual、projection、retraction 的关键矩阵乘法用 fp32。

---

## 7. Lambda 初始化

如果没有旧的 \(\Lambda_b\)，推荐 warm 初始化，而不是简单置零。

### 7.1 Column-Stiefel warm init

```python
S = W_b.T @ M_b + M_b.T @ W_b
Lambda_b = -0.25 * sym(S)
```

理由：如果暂时忽略 `msign` 的非线性，取这个值可以抵消 `M_b` 的法向分量。

### 7.2 Row-Stiefel warm init

```python
S = W_b @ M_b.T + M_b @ W_b.T
Lambda_b = -0.25 * sym(S)
```

### 7.3 零初始化 fallback

为了 ablation，可以支持：

```python
Lambda_b = torch.zeros(lambda_shape, dtype=torch.float32, device=W_b.device)
```

---

## 8. 必须实现的 helper functions

下面是建议让 Codex 添加或改造的 helper。命名可以按现有代码风格调整。

### 8.1 sym

```python
def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))
```

---

### 8.2 block_normal_correction

```python
def block_normal_correction(
    Wb: torch.Tensor,
    Lambdab: torch.Tensor,
    orientation: str,
) -> torch.Tensor:
    """
    orientation:
      - "column": Wb.T @ Wb = I
      - "row":    Wb @ Wb.T = I
    returns normal correction with same shape as Wb.
    """
    if orientation == "column":
        return 2.0 * (Wb @ Lambdab)
    if orientation == "row":
        return 2.0 * (Lambdab @ Wb)
    raise ValueError(f"unknown orientation: {orientation}")
```

---

### 8.3 block_tangent_residual

```python
def block_tangent_residual(
    Wb: torch.Tensor,
    Pb: torch.Tensor,
    orientation: str,
) -> torch.Tensor:
    """
    Returns symmetric residual H_b.
    H_b == 0 means Pb is tangent at Wb.
    """
    if orientation == "column":
        return sym(Wb.transpose(-1, -2) @ Pb + Pb.transpose(-1, -2) @ Wb)
    if orientation == "row":
        return sym(Wb @ Pb.transpose(-1, -2) + Pb @ Wb.transpose(-1, -2))
    raise ValueError(f"unknown orientation: {orientation}")
```

Strictly speaking, the expressions are already symmetric in exact arithmetic, but `sym(...)` is still useful in fp32/bf16 training.

---

### 8.4 block_tangent_project

```python
def block_tangent_project(
    Wb: torch.Tensor,
    Pb: torch.Tensor,
    orientation: str,
) -> torch.Tensor:
    """
    Projects Pb to tangent space at Wb.
    """
    if orientation == "column":
        A = sym(Wb.transpose(-1, -2) @ Pb)
        return Pb - Wb @ A
    if orientation == "row":
        A = sym(Pb @ Wb.transpose(-1, -2))
        return Pb - A @ Wb
    raise ValueError(f"unknown orientation: {orientation}")
```

---

### 8.5 global_spectral_norm_cap

After final block projection, cap the packed update by global spectral norm.

```python
@torch.no_grad()
def global_spectral_norm_cap(
    D: torch.Tensor,
    cap: float = 1.0,
    power_iters: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Estimate ||D||_2 and rescale D if needed.
    For tensors with more than 2 dims, use the same flattening convention
    as the existing Muon implementation.
    """
    D2 = D.float()
    if D2.ndim != 2:
        D2 = D2.reshape(D2.shape[0], -1)

    # Simple stateless power iteration. It is OK to replace this with
    # cached u/v vectors if the existing optimizer already has them.
    v = torch.randn(D2.shape[1], device=D2.device, dtype=D2.dtype)
    v = v / (v.norm() + eps)
    for _ in range(power_iters):
        u = D2 @ v
        u = u / (u.norm() + eps)
        v = D2.transpose(-1, -2) @ u
        v = v / (v.norm() + eps)
    sigma = (D2 @ v).norm()

    scale = torch.clamp(sigma / cap, min=1.0)
    return D / scale.to(dtype=D.dtype)
```

If the current Muon implementation already applies a shape-dependent scaling after `msign`, keep the existing convention, but avoid applying both an incompatible blockwise scaling and a global scaling.

---

## 9. Main algorithm pseudocode

This assumes existing code already has:

- momentum update producing `M`;
- `global_msign(Z)` or equivalent Newton-Schulz polar routine;
- block slicing / packing utilities;
- blockwise retraction.

### 9.1 Warm-start 1-call implementation

```python
@torch.no_grad()
def block_dual_corrected_global_muon_direction(
    W: torch.Tensor,
    M: torch.Tensor,
    block_specs: list,
    state: dict,
    global_msign,
    *,
    dual_lr: float = 0.1,
    dual_init: str = "warm",
    final_tangent_projection: bool = True,
    final_global_spectral_cap: bool = True,
    spectral_cap_value: float = 1.0,
    spectral_norm_power_iters: int = 1,
) -> torch.Tensor:
    """
    Returns D with same shape as W.

    Important:
      - global_msign is called on the packed full matrix Z, not per block.
      - block_specs must preserve the same layout as W.
    """
    W32 = W.float()
    M32 = M.float()

    W_blocks = unpack_blocks(W32, block_specs)
    M_blocks = unpack_blocks(M32, block_specs)

    lambdas = state.setdefault("block_dual_lambdas", [])
    if len(lambdas) == 0:
        for Wb, Mb, spec in zip(W_blocks, M_blocks, block_specs):
            orientation = spec.orientation  # "column" or "row"
            if dual_init == "warm":
                if orientation == "column":
                    S = Wb.T @ Mb + Mb.T @ Wb
                    L = -0.25 * sym(S)
                elif orientation == "row":
                    S = Wb @ Mb.T + Mb @ Wb.T
                    L = -0.25 * sym(S)
                else:
                    raise ValueError(f"unknown orientation: {orientation}")
            elif dual_init == "zero":
                if orientation == "column":
                    L = torch.zeros(Wb.shape[1], Wb.shape[1], device=Wb.device, dtype=torch.float32)
                elif orientation == "row":
                    L = torch.zeros(Wb.shape[0], Wb.shape[0], device=Wb.device, dtype=torch.float32)
                else:
                    raise ValueError(f"unknown orientation: {orientation}")
            else:
                raise ValueError(f"unknown dual_init: {dual_init}")
            lambdas.append(L)

    # 1. Build corrected global Muon input.
    Z_blocks = []
    for Wb, Mb, L, spec in zip(W_blocks, M_blocks, lambdas, block_specs):
        L = sym(L.float())
        Zb = Mb + block_normal_correction(Wb, L, spec.orientation)
        Z_blocks.append(Zb)

    Z = pack_blocks_like(Z_blocks, W32, block_specs)

    # 2. One global msign / polar. Do not do this per block.
    P = global_msign(Z)
    P32 = P.float()
    P_blocks = unpack_blocks(P32, block_specs)

    # 3. Update Lambda for the next optimizer step.
    new_lambdas = []
    residual_stats = []
    for Wb, Pb, L, spec in zip(W_blocks, P_blocks, lambdas, block_specs):
        H = block_tangent_residual(Wb, Pb, spec.orientation)
        L_new = sym(L.float() - dual_lr * H)
        new_lambdas.append(L_new.detach())
        residual_stats.append(H.norm().detach())
    state["block_dual_lambdas"] = new_lambdas
    state["block_dual_last_residual_norms"] = residual_stats

    # 4. Final safe tangent projection.
    if final_tangent_projection:
        D_blocks = []
        for Wb, Pb, spec in zip(W_blocks, P_blocks, block_specs):
            Db = block_tangent_project(Wb, Pb, spec.orientation)
            D_blocks.append(Db)
        D = pack_blocks_like(D_blocks, W32, block_specs)
    else:
        D = P32

    # 5. Preserve global spectral budget after projection.
    if final_global_spectral_cap:
        D = global_spectral_norm_cap(
            D,
            cap=spectral_cap_value,
            power_iters=spectral_norm_power_iters,
        )

    return D.to(dtype=W.dtype)
```

### 9.2 Optional 2-call refinement

```python
@torch.no_grad()
def block_dual_corrected_global_muon_direction_refine_2call(...):
    # Same initialization as 1-call version.

    # First call.
    Z = build_corrected_Z(W_blocks, M_blocks, lambdas, block_specs)
    P = global_msign(Z).float()
    P_blocks = unpack_blocks(P, block_specs)

    # One in-step dual correction.
    for i, (Wb, Pb, L, spec) in enumerate(zip(W_blocks, P_blocks, lambdas, block_specs)):
        H = block_tangent_residual(Wb, Pb, spec.orientation)
        lambdas[i] = sym(L.float() - dual_lr * H).detach()

    # Second call with corrected lambdas.
    Z = build_corrected_Z(W_blocks, M_blocks, lambdas, block_specs)
    P = global_msign(Z).float()
    P_blocks = unpack_blocks(P, block_specs)

    # Update lambdas again for next step, optional but useful.
    for i, (Wb, Pb, L, spec) in enumerate(zip(W_blocks, P_blocks, lambdas, block_specs)):
        H = block_tangent_residual(Wb, Pb, spec.orientation)
        lambdas[i] = sym(L.float() - dual_lr * H).detach()

    # Then final tangent projection + global spectral cap.
```

---

## 10. Integration instructions for Codex

Find the current code path that does something like:

```python
M = update_momentum(...)
P = global_msign(M)
for block in blocks:
    D_b = tangent_project(W_b, P_b)
    W_b = retract(W_b - lr * D_b)
```

Replace only the direction construction part:

```python
P = global_msign(M)
for block in blocks:
    D_b = tangent_project(W_b, P_b)
```

with:

```python
D = block_dual_corrected_global_muon_direction(
    W=W,
    M=M,
    block_specs=block_specs,
    state=state[p],
    global_msign=global_msign,
    dual_lr=group.get("block_dual_lr", 0.1),
    dual_init=group.get("block_dual_init", "warm"),
    final_tangent_projection=True,
    final_global_spectral_cap=True,
    spectral_cap_value=group.get("spectral_cap_value", 1.0),
    spectral_norm_power_iters=group.get("spectral_norm_power_iters", 1),
)

for block in blocks:
    D_b = slice_block(D, block)
    W_b = retract(W_b - lr * D_b)
```

Important constraints:

1. `global_msign` must receive the full packed matrix `Z`, not individual `Z_b` blocks.
2. Do not add blockwise `msign` anywhere in this path.
3. Keep the existing optimizer sign convention. If current code updates `W -= lr * P`, then use `W -= lr * D`. If current code uses `W += lr * P`, preserve that convention.
4. Use fp32 for dual correction, residual, projection, and retraction internals.
5. Convert back to parameter dtype only at the final assignment.

---

## 11. Shape and orientation handling

The safest implementation is to make block orientation explicit in `block_specs`.

Example:

```python
@dataclass
class BlockSpec:
    row_slice: slice
    col_slice: slice
    orientation: Literal["column", "row"]
```

Recommended default rule:

```python
if Wb.shape[0] >= Wb.shape[1]:
    orientation = "column"  # Wb.T @ Wb = I
else:
    orientation = "row"     # Wb @ Wb.T = I
```

But if the existing code already defines whether a block is row- or column-orthogonal, follow the existing definition. Do not infer a different orientation silently.

---

## 12. Retraction

After `D` is produced, the update should still use the existing blockwise retraction.

Column-Stiefel:

```python
Yb = Wb - lr * Db
Wb_new = polar_retraction_column(Yb)  # enforce Wb_new.T @ Wb_new ~= I
```

Row-Stiefel:

```python
Yb = Wb - lr * Db
Wb_new = polar_retraction_row(Yb)     # enforce Wb_new @ Wb_new.T ~= I
```

A simple row-Stiefel retraction can be implemented by transposing into column form:

```python
def polar_retraction_row(Y):
    return polar_retraction_column(Y.T).T
```

Use the current project’s preferred retraction if it already exists.

---

## 13. Metrics to log

Add lightweight diagnostics. These are important for verifying whether the dual correction is doing anything useful.

### 13.1 Tangent residual before final projection

For each block:

```python
H_b = block_tangent_residual(W_b, P_b, orientation)
r_tan_b = H_b.norm() / math.sqrt(H_b.numel())
```

Log:

```python
block_dual/tangent_residual_mean
block_dual/tangent_residual_max
```

### 13.2 Orthogonality residual after retraction

Column-Stiefel:

```python
R_b = W_b.T @ W_b - I
```

Row-Stiefel:

```python
R_b = W_b @ W_b.T - I
```

Log:

```python
block_dual/orth_residual_mean
block_dual/orth_residual_max
```

### 13.3 Global and block update norms

Log:

```python
block_dual/global_update_spectral_norm
block_dual/global_update_fro_norm
block_dual/max_block_spectral_norm
block_dual/max_lambda_norm
```

Expected behavior:

- `tangent_residual_*` before final projection should be lower than the baseline `msign(M)` residual after a few optimizer steps.
- `orth_residual_*` after retraction should stay small.
- `global_update_spectral_norm` should be at or below `spectral_cap_value` after cap.

---

## 14. Tests Codex should add

### 14.1 Column-Stiefel projection test

Create random tall `W`, project it to Stiefel, random `P`, then:

```python
D = block_tangent_project(W, P, "column")
residual = W.T @ D + D.T @ W
assert residual.norm() < tolerance
```

### 14.2 Row-Stiefel projection test

Create random wide `W`, project it to row-Stiefel, random `P`, then:

```python
D = block_tangent_project(W, P, "row")
residual = W @ D.T + D @ W.T
assert residual.norm() < tolerance
```

### 14.3 No blockwise msign regression test

Monkeypatch or count calls to `global_msign`:

- `warmstart_1call` should call `global_msign` exactly once per parameter update.
- `refine_2call` should call it exactly twice.
- It should never call `global_msign` once per block.

### 14.4 Spectral cap test

For random packed `D`:

```python
D_capped = global_spectral_norm_cap(D, cap=1.0, power_iters=3)
assert estimated_spectral_norm(D_capped) <= 1.01
```

### 14.5 Lambda state test

After one optimizer step:

```python
assert "block_dual_lambdas" in state[p]
for L in state[p]["block_dual_lambdas"]:
    assert torch.allclose(L, L.T, atol=1e-5, rtol=1e-5)
    assert L.dtype == torch.float32
```

---

## 15. Ablations to run

Run at least these four variants:

| Variant | Muon polar 粒度 | Stiefel 约束粒度 | 说明 |
|---|---:|---:|---|
| Baseline A | global | block | 当前方法：global `msign(M)` 后 block projection |
| Baseline B | block | block | 不推荐，但作为对照：每个 block 独立 Muon |
| Proposed A | global | block | warm-start 1-call block-dual correction |
| Proposed B | global | block | 2-call refinement block-dual correction |

主要比较：

```text
loss stability
loss spike frequency
validation loss
block orthogonality residual
pre-projection tangent residual
global update spectral norm
per-head or per-block activation/logit outliers
```

---

## 16. Common failure modes

### 16.1 Loss spike 或 update 变大

优先检查：

1. 是否误用了 blockwise `msign`。
2. 是否 final projection 后没有 global spectral norm cap。
3. 是否同时套用了 full-Muon scaling 和 block-Muon scaling。
4. `block_dual_lr` 是否太大。
5. row/column orientation 是否搞反。

修复顺序：

```python
block_dual_lr = 0.05
final_global_spectral_cap = True
spectral_cap_value = 1.0
block_dual_mode = "warmstart_1call"
```

### 16.2 Tangent residual 没有下降

可能原因：

1. `Lambda` 没有 warm start 或没有持久化到 optimizer state。
2. dual update sign 反了。
3. `global_msign` 内部改变了布局或转置规则，导致 pack/unpack 不一致。

排查方式：

- 记录 `H.norm()` 在 dual update 前后的变化。
- 如果 residual 持续变大，尝试把：

```python
L_new = L - dual_lr * H
```

临时改成：

```python
L_new = L + dual_lr * H
```

如果改号后明显下降，说明当前代码里的 `M` / update direction sign convention 与本文档公式相反。最终实现应与现有 optimizer 的 sign convention 保持一致。

### 16.3 Orthogonality residual 漂移

可能原因：

1. retraction 没有用 fp32。
2. row-Stiefel 和 column-Stiefel retraction 混用。
3. 参数更新后没有把 block 写回原 tensor 的正确 slice。
4. 分布式 / tensor-parallel 场景下，每个 shard 只做了局部 retraction，而 constraint 期望的是跨 shard block。

---

## 17. Minimal diff 目标

Codex 优先做 minimal diff，不要重构整个 optimizer。

最小可接受改动：

1. 增加 `block_dual_lambdas` state。
2. 增加 helper：
   - `sym`
   - `block_normal_correction`
   - `block_tangent_residual`
   - `block_tangent_project`
   - `global_spectral_norm_cap`
3. 在当前 global Muon direction 计算前插入 normal correction：
   ```python
   Z_b = M_b + normal_correction(W_b, Lambda_b)
   Z = pack(Z_b)
   P = global_msign(Z)
   ```
4. 用 residual 更新 `Lambda_b`。
5. 保留 final block tangent projection。
6. 加 final global spectral cap。
7. 保留原来的 blockwise retraction。

---

## 18. 最重要的实现原则

请让 Codex 严格遵守这几条：

1. **只在完整 packed 大矩阵上做 Muon polar / matrix sign。**
2. **不要在每个小 block 上独立做 Muon。**
3. **block 只负责 Stiefel tangent residual、dual variable、projection 和 retraction。**
4. **final tangent projection 必须保留，dual correction 只是让 projection 前的方向更好。**
5. **final global spectral norm cap 建议保留，避免 block projection 改变有效 update scale。**
6. **Lambda、projection、retraction 尽量使用 fp32。**
7. **保留现有 optimizer 的 sign convention，不要同时改 sign、learning rate 和 scaling。**

