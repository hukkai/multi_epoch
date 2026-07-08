# 1.3B 4096L Speed Check Runs

Run each config for the same wall-clock window, then compare `tokens_per_second`
from each output directory's `metrics.jsonl`.

```bash
# Baseline dense AdamW
bash run.sh configs/1p3b_4096l/speed_check/00_dense_adamw.yaml

# Current grouped_matrix index access
bash run.sh configs/1p3b_4096l/speed_check/01_grouped_no_chunk_affine_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/02_grouped_chunk_affine_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/03_grouped_no_chunk_affine_orth_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/04_grouped_chunk_affine_muon.yaml
bash run.sh configs/1p3b_4096l/speed_check/05_grouped_chunk_affine_orth_muon.yaml

# AdamW-only index vs unbind comparison
bash run.sh configs/1p3b_4096l/speed_check/06_grouped_no_chunk_affine_adamw_unbind.yaml
bash run.sh configs/1p3b_4096l/speed_check/07_grouped_chunk_affine_adamw_unbind.yaml
```

Suggested comparisons:

- `00` vs `01`: dense AdamW vs grouped_matrix AdamW without chunk affine
- `01` vs `02`: chunk affine off vs on under grouped_matrix AdamW
- `01` vs `06`: index vs unbind without chunk affine
- `02` vs `07`: index vs unbind with chunk affine
- `04` vs `05`: Muon vs OrthMuon with chunk affine
