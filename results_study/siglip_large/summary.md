# Connector study results


## Test, selected by validation worst-group accuracy

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck       |       2 | 89.6 ± 0.5 | 79.3 ± 1.9 | 97.9 ± 0.3 | 81.2 ± 0.7 | 0.2 ± 0.0 | 88.4 ± 0.7 | 89.6 ± 0.5 |                    79.6 |
| waterbirds | bottleneck_fixed |       2 | 89.8 ± 1.8 | 80.1 ± 3.1 | 98.9 ± 0.2 | 80.8 ± 3.5 | 0.2 ± 0.0 | 90.0 ± 2.1 | 89.8 ± 1.8 |                    80   |
| waterbirds | mlp              |       2 | 92.0 ± 0.6 | 82.9 ± 3.4 | 99.2 ± 0.3 | 84.9 ± 1.4 | 0.1 ± 0.0 | 91.7 ± 1.3 | 92.0 ± 0.6 |                    83   |
| waterbirds | qformer          |       2 | 90.9 ± 0.7 | 81.7 ± 0.4 | 98.8 ± 0.3 | 83.1 ± 1.7 | 0.2 ± 0.0 | 90.5 ± 0.1 | 90.9 ± 0.7 |                    81.7 |



Paired worst-group difference vs MLP (wga selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck − mlp       |       2 | -3.7 ± 5.3  |
| waterbirds | bottleneck_fixed − mlp |       2 | -2.8 ± 0.3  |
| waterbirds | qformer − mlp          |       2 | -1.2 ± 3.0  |



## Test, selected by validation average accuracy (no group labels)

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck       |       2 | 92.1 ± 1.4 | 73.4 ± 0.8 | 97.9 ± 0.1 | 86.3 ± 2.8 | 0.1 ± 0.0 | 88.8 ± 1.0 | 92.1 ± 1.4 |                    74.5 |
| waterbirds | bottleneck_fixed |       2 | 93.1 ± 1.3 | 80.5 ± 6.4 | 98.8 ± 0.1 | 87.3 ± 2.5 | 0.1 ± 0.0 | 91.3 ± 2.0 | 93.1 ± 1.3 |                    80.5 |
| waterbirds | mlp              |       2 | 95.0 ± 0.9 | 80.7 ± 5.4 | 98.6 ± 0.7 | 91.5 ± 2.5 | 0.1 ± 0.0 | 92.3 ± 1.1 | 95.0 ± 0.9 |                    80.7 |
| waterbirds | qformer          |       2 | 93.7 ± 0.4 | 69.7 ± 5.4 | 98.0 ± 0.0 | 89.4 ± 0.7 | 0.1 ± 0.0 | 89.0 ± 0.8 | 93.7 ± 0.4 |                    69.5 |



Paired worst-group difference vs MLP (avg selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck − mlp       |       2 | -7.3 ± 6.2  |
| waterbirds | bottleneck_fixed − mlp |       2 | -0.3 ± 1.0  |
| waterbirds | qformer − mlp          |       2 | -11.1 ± 0.0 |

