# 1.3B Base Sweeps

This directory contains the 1.3B optimizer sweeps for the 2048 x 24 GQA G=2
model at 4096 sequence length.

- AdamW: `train.lr x train.weight_decay`
- Muon: `optim.muon_lr x optim.muon_weight_decay`
- OrthAdam: `train.lr`
- OrthMuon: `optim.muon_lr`

The sweep ranges are:

- AdamW lr: `0.0006, 0.0012, 0.002`; weight decay: `0.1, 0.3, 0.6`
- Muon lr: `0.0005, 0.001, 0.002`; weight decay: `0.1, 0.3, 0.6`
- OrthAdam lr: `0.0006, 0.0012, 0.002`
- OrthMuon lr: `0.0005, 0.001, 0.002`

Run every config listed in `all_sweeps.txt` for the full 24-run sweep.
