# 1.3B 4096L Speed Check

Run these configs for the same wall-clock window and compare token speed:

```bash
bash run.sh configs/1p3b_4096l/speed_check/00_dense_adamw.yaml

# Current grouped weight access: model.layer_weight_access=index
bash run.sh configs/1p3b_4096l/speed_check/01_grouped_no_chunk_affine_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/02_grouped_chunk_affine_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/03_grouped_no_chunk_affine_orth_adamw.yaml
bash run.sh configs/1p3b_4096l/speed_check/04_grouped_chunk_affine_muon.yaml
bash run.sh configs/1p3b_4096l/speed_check/05_grouped_chunk_affine_orth_muon.yaml

# Forward-local unbind weight access for the AdamW forward-path comparison:
bash run.sh configs/1p3b_4096l/speed_check/06_grouped_no_chunk_affine_adamw_unbind.yaml
bash run.sh configs/1p3b_4096l/speed_check/07_grouped_chunk_affine_adamw_unbind.yaml
```

Compare each index/unbind pair:

- `01` vs `06`
- `02` vs `07`
