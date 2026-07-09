# 360M 4096L Affine Ablation

Run these no-affine configs and compare against the matching existing affine
sweep configs.

```bash
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr/orth_adam_lr0p002_no_affine.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p004_wd0p5_no_affine.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p3_no_affine.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p002_no_affine.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p004_no_affine.yaml
```

Matched comparisons:

- `orth_adam_lr0p002.yaml` vs `orth_adam_lr0p002_no_affine.yaml`
- `muon_lr0p004_wd0p5.yaml` vs `muon_lr0p004_wd0p5_no_affine.yaml`
- `muon_lr0p002_wd0p3.yaml` vs `muon_lr0p002_wd0p3_no_affine.yaml`
- `orth_muon_lr0p002.yaml` vs `orth_muon_lr0p002_no_affine.yaml`
- `orth_muon_lr0p004.yaml` vs `orth_muon_lr0p004_no_affine.yaml`
