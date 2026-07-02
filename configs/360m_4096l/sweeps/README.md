# 360M Base Sweeps

This directory contains the 360M optimizer sweeps:

- AdamW: `train.lr x train.weight_decay`
- Muon: `optim.muon_lr x optim.muon_weight_decay`
- OrthAdam: `train.lr`
- OrthMuon: `optim.muon_lr`

The sweep ranges are:

- AdamW lr/weight decay: 11 selected pairs in `adamw_lr_wd/`
- Muon lr/weight decay: 12 selected pairs in `muon_lr_wd/`
- OrthAdam lr: `0.0006, 0.0012, 0.002`
- OrthMuon lr: `0.001, 0.002, 0.004, 0.008`

Run every config listed in `base_adamw_muon_lr_wd.txt` for the base AdamW/Muon
sweep. OrthAdam and OrthMuon sweeps live in `orth_adam_lr/` and `orth_muon_lr/`.
