# RUN

## Before Running

Open a terminal in this repository:

```bash
cd /Users/kaihu/Desktop/multi_epoch
```

## AdamW Base Sweep

Run these 25 AdamW experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p03_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p03_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p03_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p1_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p1_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p1_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p3_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p3_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p03_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p03_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p03_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p1_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p3_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p3_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p03_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p03_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p03_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p1_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p1_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p1_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p3_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p002_wd0p3_cos2.yaml
```

## Muon Base Sweep

Run these 25 Muon experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p03_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p03_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p03_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p1_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p1_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p1_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p3_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p001_wd0p3_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p03_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p03_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p03_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p1_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p3_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p002_wd0p3_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p03_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p03_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p03_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p1_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p1_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p1_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p3_cos1p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p3_cos2.yaml
```

## After The Sweep

This run list has 50 experiments instead of the full 54-grid. Four settings are
omitted because matching pilot runs already exist:

- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=1.0`
- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=2.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=1.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=2.0`

Select the top 5 AdamW and top 5 Muon configs by validation loss. Orth
follow-up configs should extend those selected configs, inherit the base lr
setting, and use `train.cosine_power: 2.0` as the fixed Orth recipe.
