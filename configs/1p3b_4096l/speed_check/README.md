# 1.3B 4096L Speed Check

Run these configs for the same wall-clock window and compare token speed:

```bash
bash run.sh configs/1p3b_4096l/speed_check/00_dense_adamw.yaml

# grouped_matrix now always uses forward-local unbind access
bash run.sh configs/1p3b_4096l/speed_check/01_grouped_no_chunk_affine_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/02_grouped_chunk_affine_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/03_grouped_chunk_affine_orth_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/04_grouped_chunk_affine_muon.yaml
bash run.sh configs/1p3b_4096l/speed_check/05_grouped_chunk_affine_orth_muon.yaml
```

Suggested comparisons:

- `00` vs `01`: dense AdamW vs grouped_matrix AdamW without chunk affine
- `01` vs `02`: chunk affine off vs on under grouped_matrix AdamW
- `02` vs `03`: AdamW vs OrthAdam with chunk affine
- `04` vs `05`: Muon vs OrthMuon with chunk affine
