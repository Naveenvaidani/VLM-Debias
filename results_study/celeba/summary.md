# Connector study results


## Test, selected by validation worst-group accuracy

| task   | connector        |   seeds | Avg         | WGA         | ID          | OOD        | RSR        | Bal.       | Gen.        |   Held-out template WGA |
|:-------|:-----------------|--------:|:------------|:------------|:------------|:-----------|:-----------|:-----------|:------------|------------------------:|
| celeba | bottleneck       |       2 | 82.9 ± 14.4 | 47.1 ± 19.9 | 75.8 ± 19.8 | 90.1 ± 9.1 | -0.2 ± 0.2 | 86.4 ± 4.2 | 82.9 ± 14.4 |                    38.4 |
| celeba | bottleneck_fixed |       1 | 95.0 ± 0.0  | 46.1 ± 0.0  | 93.3 ± 0.0  | 96.7 ± 0.0 | -0.0 ± 0.0 | 91.2 ± 0.0 | 95.0 ± 0.0  |                    46.1 |
| celeba | mlp              |       2 | 95.2 ± 0.0  | 42.7 ± 4.6  | 94.4 ± 0.0  | 96.1 ± 0.1 | -0.0 ± 0.0 | 90.2 ± 0.3 | 95.2 ± 0.0  |                    42.2 |



Paired worst-group difference vs MLP (wga selection):

| task   | comparison             |   seeds | ΔWGA (pp)   |
|:-------|:-----------------------|--------:|:------------|
| celeba | bottleneck − mlp       |       2 | +4.4 ± 15.4 |
| celeba | bottleneck_fixed − mlp |       1 | +0.2 ± 0.0  |



## Test, selected by validation average accuracy (no group labels)

| task   | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR        | Bal.       | Gen.       |   Held-out template WGA |
|:-------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|------------------------:|
| celeba | bottleneck       |       2 | 94.3 ± 0.4 | 18.3 ± 5.0 | 93.8 ± 1.2 | 94.8 ± 0.4 | -0.0 ± 0.0 | 86.2 ± 0.6 | 94.3 ± 0.4 |                    16.7 |
| celeba | bottleneck_fixed |       1 | 95.0 ± 0.0 | 46.1 ± 0.0 | 93.3 ± 0.0 | 96.7 ± 0.0 | -0.0 ± 0.0 | 91.2 ± 0.0 | 95.0 ± 0.0 |                    46.1 |
| celeba | mlp              |       2 | 95.2 ± 0.0 | 42.7 ± 4.6 | 94.4 ± 0.0 | 96.1 ± 0.1 | -0.0 ± 0.0 | 90.2 ± 0.3 | 95.2 ± 0.0 |                    42.2 |



Paired worst-group difference vs MLP (avg selection):

| task   | comparison             |   seeds | ΔWGA (pp)   |
|:-------|:-----------------------|--------:|:------------|
| celeba | bottleneck − mlp       |       2 | -24.4 ± 9.6 |
| celeba | bottleneck_fixed − mlp |       1 | +0.2 ± 0.0  |

