# 360M Base Sweeps

This directory contains the first-stage base optimizer sweep to run now:

- AdamW: `train.lr x train.weight_decay x train.cosine_power`
- Muon: `optim.muon_lr x optim.muon_weight_decay x train.cosine_power`

The sweep ranges are:

- AdamW lr: `0.0006, 0.0012, 0.002`
- Muon lr: `0.001, 0.002, 0.004`
- weight decay: `0.03, 0.1, 0.3`
- cosine power: `1.0, 1.5, 2.0`

The full grid has 54 settings. Four settings are omitted because matching
pilot runs already exist:

- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=1.0`
- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=2.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=1.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=2.0`

Run every config listed in `base_adamw_muon_lr_wd_cosine_power.txt`; it contains
the remaining 50 settings.
After selecting the top 5 AdamW and top 5 Muon configs by validation loss,
create OrthAdam/OrthMuon configs by extending those selected configs. The
Orth follow-up should inherit the selected lr values and use
`train.cosine_power: 2.0` as the fixed Orth recipe.
