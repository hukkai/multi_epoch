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
