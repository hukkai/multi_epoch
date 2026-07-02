# Experiment Results

This file records parameters only, not config paths or config names.
Final validation loss is lower-is-better.

## Global Best

| Optimizer | Best params | Final val |
|---|---|---:|
| AdamW | lr=0.0012, weight_decay=0.3 | 2.5509 |
| OrthAdam | lr=0.002 | 2.5066 |
| Muon | lr=0.004, weight_decay=0.5 | 2.4746 |
| OrthMuon | lr=0.002 | 2.4602 |

## Summary

| Family | LR | Base optimizer best params | Base loss | Orth best params | Orth loss | Winner |
|---|---:|---|---:|---|---:|---|
| AdamW vs OrthAdam | 0.0006 | optimizer=AdamW, weight_decay=0.6 | 2.5560 | optimizer=OrthAdam | 2.5790 | AdamW |
| AdamW vs OrthAdam | 0.0012 | optimizer=AdamW, weight_decay=0.3 | 2.5509 | optimizer=OrthAdam | 2.5169 | OrthAdam |
| AdamW vs OrthAdam | 0.002 | optimizer=AdamW, weight_decay=0.3 | 2.5617 | optimizer=OrthAdam | 2.5066 | OrthAdam |
| Muon vs OrthMuon | 0.001 | optimizer=Muon, weight_decay=0.3 | 2.4874 | optimizer=OrthMuon | 2.4843 | OrthMuon |
| Muon vs OrthMuon | 0.002 | optimizer=Muon, weight_decay=0.3 | 2.4782 | optimizer=OrthMuon | 2.4602 | OrthMuon |
| Muon vs OrthMuon | 0.004 | optimizer=Muon, weight_decay=0.5 | 2.4746 | optimizer=OrthMuon | 2.4945 | Muon |
| Muon vs OrthMuon | 0.008 | optimizer=Muon, weight_decay=0.6 | 2.8559 | optimizer=OrthMuon | 2.5770 | OrthMuon |

Takeaway: OrthMuon has the best overall result at `lr=0.002`, with final val
`2.4602`. Tuned Muon is close and wins at `lr=0.004`. OrthAdam wins at
`lr=0.0012` and `lr=0.002`, while AdamW wins at `lr=0.0006`.

## Raw Losses

### AdamW


<table>
  <thead>
    <tr>
      <th>LR</th>
      <th>Weight decay</th>
      <th>Final val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">0.0006</td>
      <td>0.03</td>
      <td>2.5914</td>
    </tr>
    <tr>
      <td>0.1</td>
      <td>2.5846</td>
    </tr>
    <tr>
      <td>0.3</td>
      <td>2.5634</td>
    </tr>
    <tr>
      <td>0.6</td>
      <td>2.5560</td>
    </tr>
    <tr>
      <td rowspan="4">0.0012</td>
      <td>0.03</td>
      <td>2.5699</td>
    </tr>
    <tr>
      <td>0.1</td>
      <td>2.5607</td>
    </tr>
    <tr>
      <td>0.3</td>
      <td>2.5509</td>
    </tr>
    <tr>
      <td>0.6</td>
      <td>2.5526</td>
    </tr>
    <tr>
      <td rowspan="3">0.002</td>
      <td>0.03</td>
      <td>2.5804</td>
    </tr>
    <tr>
      <td>0.1</td>
      <td>2.5688</td>
    </tr>
    <tr>
      <td>0.3</td>
      <td>2.5617</td>
    </tr>
  </tbody>
</table>

### Muon


<table>
  <thead>
    <tr>
      <th>LR</th>
      <th>Weight decay</th>
      <th>Final val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">0.001</td>
      <td>0.03</td>
      <td>2.5253</td>
    </tr>
    <tr>
      <td>0.1</td>
      <td>2.5075</td>
    </tr>
    <tr>
      <td>0.3</td>
      <td>2.4874</td>
    </tr>
    <tr>
      <td rowspan="3">0.002</td>
      <td>0.03</td>
      <td>2.5026</td>
    </tr>
    <tr>
      <td>0.1</td>
      <td>2.4925</td>
    </tr>
    <tr>
      <td>0.3</td>
      <td>2.4782</td>
    </tr>
    <tr>
      <td rowspan="4">0.004</td>
      <td>0.03</td>
      <td>2.5018</td>
    </tr>
    <tr>
      <td>0.1</td>
      <td>2.4906</td>
    </tr>
    <tr>
      <td>0.3</td>
      <td>2.4763</td>
    </tr>
    <tr>
      <td>0.5</td>
      <td>2.4746</td>
    </tr>
    <tr>
      <td rowspan="2">0.008</td>
      <td>0.3</td>
      <td>3.2371</td>
    </tr>
    <tr>
      <td>0.6</td>
      <td>2.8559</td>
    </tr>
  </tbody>
</table>

### OrthAdam

| LR | Final val |
|---:|---:|
| 0.0006 | 2.5790 |
| 0.0012 | 2.5169 |
| 0.002 | 2.5066 |

### OrthMuon

| LR | Final val |
|---:|---:|
| 0.001 | 2.4843 |
| 0.002 | 2.4602 |
| 0.004 | 2.4945 |
| 0.008 | 2.5770 |
