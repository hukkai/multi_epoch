# New Run


Run all 24 configs listed in `configs/1p3b_4096l/sweeps/all_sweeps.txt`.

### AdamW LR/WD Sweep

```bash
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p1.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p3.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p6.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p1.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p3.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p6.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p002_wd0p1.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p002_wd0p3.yaml
bash run.sh configs/1p3b_4096l/sweeps/adamw_lr_wd/adamw_lr0p002_wd0p6.yaml
```

### Muon LR/WD Sweep

```bash
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p0005_wd0p1.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p0005_wd0p3.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p0005_wd0p6.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p001_wd0p1.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p001_wd0p3.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p001_wd0p6.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p1.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p3.yaml
bash run.sh configs/1p3b_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p6.yaml
```

### OrthAdam LR Sweep

```bash
bash run.sh configs/1p3b_4096l/sweeps/orth_adam_lr/orth_adam_lr0p0006.yaml
bash run.sh configs/1p3b_4096l/sweeps/orth_adam_lr/orth_adam_lr0p0012.yaml
bash run.sh configs/1p3b_4096l/sweeps/orth_adam_lr/orth_adam_lr0p002.yaml
```

### OrthMuon LR Sweep

```bash
bash run.sh configs/1p3b_4096l/sweeps/orth_muon_lr/orth_muon_lr0p0005.yaml
bash run.sh configs/1p3b_4096l/sweeps/orth_muon_lr/orth_muon_lr0p001.yaml
bash run.sh configs/1p3b_4096l/sweeps/orth_muon_lr/orth_muon_lr0p002.yaml
```

## 360M Role Ablation

These configs start from the 360M Muon base config, use OrthMuon by default, and
override one role at a time to Muon.

```bash
bash run.sh configs/360m_4096l/role_ablation/attn_q_muon.yaml
bash run.sh configs/360m_4096l/role_ablation/attn_k_muon.yaml
bash run.sh configs/360m_4096l/role_ablation/attn_v_muon.yaml
bash run.sh configs/360m_4096l/role_ablation/attn_o_muon.yaml
bash run.sh configs/360m_4096l/role_ablation/mlp_gate_muon.yaml
bash run.sh configs/360m_4096l/role_ablation/mlp_up_muon.yaml
bash run.sh configs/360m_4096l/role_ablation/mlp_down_muon.yaml
```

## Total

- 1.3B base sweep: 24 runs
- 360M role ablation: 7 runs
- Total: 31 runs

## 360M Affine Ablation

Each optimizer setting tests no affine, MLP affine only, and both MLP and
attention affine. The Muon runs use the best weight decay for each learning
rate from `experience.md`.

```bash
bash run.sh configs/360m_4096l/affine_ablation/muon_lr0p002_wd0p3_no_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/muon_lr0p002_wd0p3_mlp_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/muon_lr0p002_wd0p3_all_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/muon_lr0p004_wd0p5_no_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/muon_lr0p004_wd0p5_mlp_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/muon_lr0p004_wd0p5_all_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p002_no_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p002_mlp_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p002_all_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p004_no_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p004_mlp_affine.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p004_all_affine.yaml
```

### Affine LR Multiplier

```bash
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p002_all_affine_lr_multiplier2.yaml
bash run.sh configs/360m_4096l/affine_ablation/orth_muon_lr0p004_all_affine_lr_multiplier2.yaml
```


### Submat Dim Ablation

```bash
bash run.sh configs/360m_4096l/submat_ablation/attn_muon_mlp_orth_muon_submat2.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_muon_mlp_orth_muon_submat8.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_muon_mlp_orth_muon_submat32.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_muon_mlp_orth_muon_submat128.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_muon_submat2.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_muon_submat8.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_muon_submat32.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_muon_submat128.yaml
```

```bash
bash run.sh configs/360m_4096l/submat_ablation/attn_muon_mlp_orth_muon_submat1.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_muon_mlp_orth_muon_submat16.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_muon_submat1.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_muon_submat16.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_orth_muon_submat1.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_orth_muon_submat8.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_orth_muon_submat16.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_orth_muon_mlp_orth_muon_submat128.yaml
```

### OrthMuon Landing Frequency

All three runs use `submat_dim: 32`. The strict run retracts every step; the
other runs take ambient Muon steps and land every 2 or 4 completed steps.

```bash
bash run.sh configs/360m_4096l/landing/orth_muon_strict_submat32.yaml
bash run.sh configs/360m_4096l/landing/orth_muon_land_every2_submat32.yaml
bash run.sh configs/360m_4096l/landing/orth_muon_land_every4_submat32.yaml
```

All three runs below use `submat_dim: 64`.

```bash
bash run.sh configs/360m_4096l/landing/orth_muon_strict_submat64.yaml
bash run.sh configs/360m_4096l/landing/orth_muon_land_every2_submat64.yaml
bash run.sh configs/360m_4096l/landing/orth_muon_land_every4_submat64.yaml
```
