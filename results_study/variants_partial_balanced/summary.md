# Connector study results


## Test, selected by validation worst-group accuracy

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       1 | 91.4 ± 0.0 | 87.3 ± 0.0 | 95.5 ± 0.0 | 87.4 ± 0.0 | 0.1 ± 0.0 | 90.9 ± 0.0 | 91.4 ± 0.0 |                    87.4 |
| waterbirds | mlp              |       2 | 92.1 ± 2.6 | 83.9 ± 1.9 | 96.9 ± 0.1 | 87.3 ± 5.2 | 0.1 ± 0.1 | 91.4 ± 0.9 | 92.1 ± 2.6 |                    84   |



Paired worst-group difference vs MLP (wga selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck_fixed − mlp |       1 | +2.1 ± 0.0  |



## Test, selected by validation average accuracy (no group labels)

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       1 | 91.5 ± 0.0 | 75.6 ± 0.0 | 97.7 ± 0.0 | 85.4 ± 0.0 | 0.1 ± 0.0 | 88.8 ± 0.0 | 91.5 ± 0.0 |                    75.4 |
| waterbirds | mlp              |       2 | 94.0 ± 0.2 | 80.4 ± 6.8 | 96.8 ± 0.0 | 91.2 ± 0.3 | 0.1 ± 0.0 | 90.9 ± 1.6 | 94.0 ± 0.2 |                    80.4 |



Paired worst-group difference vs MLP (avg selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck_fixed − mlp |       1 | -9.6 ± 0.0  |

