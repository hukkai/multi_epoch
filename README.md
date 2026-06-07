# ortho_llm

tag: 1.5B

baseline:
```
bash run.sh configs/1.5B/ortho_all_50k.yaml
```

```
bash run.sh configs/1.5B/adamw_50k.yaml
```


tag: "so-method"

baseline is exp 4537

ablation:
```
bash run.sh configs/0.5B/ortho_all_20k_n8_geo.yaml
```

```
bash run.sh configs/0.5B/ortho_all_20k_n8_scalar.yaml
```

