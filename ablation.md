# Ablation Runs

## 360M OrthMuon Submat Ablation

These runs extend `configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p002.yaml`
and ablate `submat_dim_overrides` around the default `submat_dim: 32`.

Baseline is `configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p002.yaml`.

```bash
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p002.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_submat_x2.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_submat_div2.yaml
bash run.sh configs/360m_4096l/submat_ablation/mlp_submat_x2.yaml
bash run.sh configs/360m_4096l/submat_ablation/mlp_submat_div2.yaml
bash run.sh configs/360m_4096l/submat_ablation/attn_submat_x4.yaml
bash run.sh configs/360m_4096l/submat_ablation/mlp_submat_x4.yaml
```
