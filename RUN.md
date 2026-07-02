# RUN

## Before Running

Open a terminal in this repository:

```bash
cd /Users/kaihu/Desktop/multi_epoch
```

## AdamW Base Sweep

Run these 11 AdamW experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p03.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p3.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0006_wd0p6.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p03.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p3.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p0012_wd0p6.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p002_wd0p03.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p002_wd0p1.yaml
bash run.sh configs/360m_4096l/sweeps/adamw_lr_wd/adamw_lr0p002_wd0p3.yaml
```

## Muon Base Sweep

Run these 12 Muon experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p001_wd0p03.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p001_wd0p1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p001_wd0p3.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p03.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p002_wd0p3.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p004_wd0p03.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p004_wd0p1.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p004_wd0p3.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p004_wd0p5.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p008_wd0p3.yaml
bash run.sh configs/360m_4096l/sweeps/muon_lr_wd/muon_lr0p008_wd0p6.yaml
```

## After The Sweep

The base sweep run list has 23 AdamW/Muon experiments. The same paths are also
listed in `configs/360m_4096l/sweeps/base_adamw_muon_lr_wd.txt`.

## OrthAdam LR Sweep

Run these 3 OrthAdam experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr/orth_adam_lr0p0006.yaml
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr/orth_adam_lr0p0012.yaml
bash run.sh configs/360m_4096l/sweeps/orth_adam_lr/orth_adam_lr0p002.yaml
```

## OrthMuon LR Sweep

Run these 4 OrthMuon experiments:

```bash
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p001.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p002.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p004.yaml
bash run.sh configs/360m_4096l/sweeps/orth_muon_lr/orth_muon_lr0p008.yaml
```
