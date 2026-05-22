# ortho_llm

0. baseline

```
bash run.sh none
```

```
bash run.sh all 64
```

1. ablation 1

```
bash run.sh all 256
```

```
bash run.sh all 16
```

2. ablation 2

```
bash run.sh mlp 64
```

```
bash run.sh atten 64
```

3. ablation 3


```
bash run.sh all 64 0.5
```

```
bash run.sh all 64 2.0
```

4. ablation 4

```
bash run.sh all 64 1.0 true false
```

```
bash run.sh all 64 1.0 false true
```

```
bash run.sh all 64 1.0 false false
```