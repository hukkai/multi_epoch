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

## Added Experiments for the Sweep

Run these 5 targeted follow-up experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0012_wd0p6_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd_cosine_power/adamw_lr0p0006_wd0p6_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p008_wd0p3_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p004_wd0p5_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd_cosine_power/muon_lr0p008_wd0p6_cos1.yaml
```

## After The Sweep

The base sweep run list has 50 experiments instead of the full 54-grid. Four
settings are omitted because matching pilot runs already exist:

- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=1.0`
- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=2.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=1.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=2.0`

With the five added experiments above, the AdamW/Muon sweep sections list 55 run
commands before the OrthAdam/OrthMuon LR sweeps below.

## OrthAdam LR Sweep

Run these 3 OrthAdam experiments with `train.cosine_power: 2`:

```bash
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr_cos2/orth_adam_lr0p0006_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr_cos2/orth_adam_lr0p0012_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr_cos2/orth_adam_lr0p002_cos2.yaml
```

## OrthMuon LR Sweep

Run these 4 OrthMuon experiments with `train.cosine_power: 2`:

```bash
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos2/orth_muon_lr0p001_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos2/orth_muon_lr0p002_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos2/orth_muon_lr0p004_cos2.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos2/orth_muon_lr0p008_cos2.yaml
```

## OrthAdam LR Sweep Cos1

Run these 3 additional OrthAdam experiments with `train.cosine_power: 1.0`:

```bash
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr_cos1/orth_adam_lr0p0006_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr_cos1/orth_adam_lr0p0012_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr_cos1/orth_adam_lr0p002_cos1.yaml
```

## OrthMuon LR Sweep Cos1

Run these 4 additional OrthMuon experiments with `train.cosine_power: 1.0`:

```bash
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos1/orth_muon_lr0p001_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos1/orth_muon_lr0p002_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos1/orth_muon_lr0p004_cos1.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr_cos1/orth_muon_lr0p008_cos1.yaml
```
