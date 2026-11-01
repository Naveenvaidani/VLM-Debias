# Connector study results


## Test, selected by validation worst-group accuracy

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       1 | 75.5 ± 0.0 | 44.5 ± 0.0 | 98.4 ± 0.0 | 52.7 ± 0.0 | 0.5 ± 0.0 | 80.4 ± 0.0 | 75.5 ± 0.0 |                    42.4 |
| waterbirds | mlp              |       1 | 90.6 ± 0.0 | 79.1 ± 0.0 | 97.7 ± 0.0 | 83.4 ± 0.0 | 0.1 ± 0.0 | 89.0 ± 0.0 | 90.6 ± 0.0 |                    79.3 |



Paired worst-group difference vs MLP (wga selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck_fixed − mlp |       1 | -34.6 ± 0.0 |



## Test, selected by validation average accuracy (no group labels)

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       1 | 88.6 ± 0.0 | 33.4 ± 0.0 | 97.8 ± 0.0 | 79.3 ± 0.0 | 0.2 ± 0.0 | 79.0 ± 0.0 | 88.6 ± 0.0 |                    34.4 |
| waterbirds | mlp              |       1 | 91.0 ± 0.0 | 67.4 ± 0.0 | 98.6 ± 0.0 | 83.4 ± 0.0 | 0.2 ± 0.0 | 87.5 ± 0.0 | 91.0 ± 0.0 |                    67.4 |



Paired worst-group difference vs MLP (avg selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck_fixed − mlp |       1 | -34.0 ± 0.0 |

