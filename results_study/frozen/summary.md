# Connector study results


## Test, selected by validation worst-group accuracy

| task       | connector   |   seeds | Avg        | WGA        | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:------------|--------:|:-----------|:-----------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck  |       3 | 75.2 ± 7.2 | 24.2 ± 2.8 | 92.6 ± 4.0 | 57.7 ± 17.9 | 0.4 ± 0.2 | 66.5 ± 1.9 | 75.0 ± 7.0 |                     5.9 |
| waterbirds | mlp         |       3 | 88.9 ± 0.8 | 73.4 ± 1.1 | 98.4 ± 0.4 | 79.4 ± 1.6  | 0.2 ± 0.0 | 87.1 ± 0.3 | 88.9 ± 0.8 |                    75   |
| waterbirds | qformer     |       3 | 89.4 ± 1.0 | 74.6 ± 0.9 | 98.4 ± 0.2 | 80.4 ± 2.3  | 0.2 ± 0.0 | 87.6 ± 0.7 | 89.4 ± 1.0 |                    72   |



Paired worst-group difference vs MLP (wga selection):

| task       | comparison       |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------|--------:|:------------|
| waterbirds | bottleneck − mlp |       3 | -49.2 ± 3.5 |
| waterbirds | qformer − mlp    |       3 | +1.2 ± 1.3  |



## Test, selected by validation average accuracy (no group labels)

| task       | connector   |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck  |       3 | 79.5 ± 4.5 | 23.1 ± 16.4 | 95.7 ± 2.4 | 63.2 ± 11.1 | 0.3 ± 0.1 | 69.9 ± 4.6 | 77.9 ± 2.1 |                    17.3 |
| waterbirds | mlp         |       3 | 89.7 ± 1.6 | 73.5 ± 2.8  | 98.9 ± 0.4 | 80.6 ± 3.6  | 0.2 ± 0.0 | 88.0 ± 0.5 | 89.7 ± 1.7 |                    74.1 |
| waterbirds | qformer     |       3 | 90.1 ± 0.5 | 66.1 ± 7.6  | 98.3 ± 0.3 | 82.0 ± 1.0  | 0.2 ± 0.0 | 86.4 ± 1.5 | 90.1 ± 0.5 |                    62.5 |



Paired worst-group difference vs MLP (avg selection):

| task       | comparison       |   seeds | ΔWGA (pp)    |
|:-----------|:-----------------|--------:|:-------------|
| waterbirds | bottleneck − mlp |       3 | -50.4 ± 18.0 |
| waterbirds | qformer − mlp    |       3 | -7.4 ± 4.9   |

