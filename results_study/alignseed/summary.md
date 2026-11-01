# Connector study results


## Test, selected by validation worst-group accuracy

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       3 | 88.6 ± 0.5 | 74.2 ± 0.8 | 98.2 ± 0.1 | 79.0 ± 1.0 | 0.2 ± 0.0 | 87.1 ± 0.2 | 88.6 ± 0.5 |                    73.9 |
| waterbirds | mlp              |       3 | 89.8 ± 1.8 | 76.1 ± 2.1 | 99.1 ± 0.1 | 80.5 ± 3.6 | 0.2 ± 0.0 | 88.7 ± 0.8 | 89.8 ± 1.8 |                    75.9 |



Paired worst-group difference vs MLP (wga selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck_fixed − mlp |       3 | -2.0 ± 1.9  |



## Test, selected by validation average accuracy (no group labels)

| task       | connector        |   seeds | Avg        | WGA         | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:------------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       3 | 90.6 ± 1.5 | 67.9 ± 8.5  | 98.1 ± 0.3 | 83.2 ± 3.3 | 0.2 ± 0.0 | 86.9 ± 1.9 | 90.6 ± 1.5 |                    67.9 |
| waterbirds | mlp              |       3 | 92.0 ± 0.8 | 64.5 ± 10.9 | 98.3 ± 0.7 | 85.7 ± 1.6 | 0.1 ± 0.0 | 87.3 ± 2.7 | 92.0 ± 0.8 |                    63.7 |



Paired worst-group difference vs MLP (avg selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck_fixed − mlp |       3 | +3.4 ± 3.8  |

