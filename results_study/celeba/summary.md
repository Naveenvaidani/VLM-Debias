# Connector study results


## Test, selected by validation worst-group accuracy

| task   | connector        |   seeds | Avg        | WGA         | ID          | OOD        | RSR        | Bal.       | Gen.       |   Held-out template WGA |
|:-------|:-----------------|--------:|:-----------|:------------|:------------|:-----------|:-----------|:-----------|:-----------|------------------------:|
| celeba | bottleneck       |       4 | 87.4 ± 9.8 | 36.3 ± 18.7 | 84.4 ± 15.4 | 90.3 ± 6.1 | -0.1 ± 0.2 | 82.6 ± 7.7 | 87.4 ± 9.8 |                    27.8 |
| celeba | bottleneck_fixed |       4 | 94.5 ± 0.5 | 53.1 ± 5.8  | 92.1 ± 1.3  | 96.9 ± 0.4 | -0.1 ± 0.0 | 91.8 ± 0.7 | 94.5 ± 0.5 |                    53.5 |
| celeba | mlp              |       4 | 94.1 ± 1.4 | 54.3 ± 13.7 | 91.5 ± 3.5  | 96.6 ± 0.7 | -0.1 ± 0.0 | 91.6 ± 1.7 | 94.1 ± 1.4 |                    54.3 |



Paired worst-group difference vs MLP (wga selection):

| task   | comparison             |   seeds | ΔWGA (pp)    |
|:-------|:-----------------------|--------:|:-------------|
| celeba | bottleneck − mlp       |       4 | -18.0 ± 28.3 |
| celeba | bottleneck_fixed − mlp |       3 | -8.6 ± 7.7   |



## Test, selected by validation average accuracy (no group labels)

| task   | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR        | Bal.       | Gen.       |   Held-out template WGA |
|:-------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|------------------------:|
| celeba | bottleneck       |       4 | 94.3 ± 0.3 | 20.9 ± 4.3 | 93.6 ± 0.8 | 95.0 ± 0.4 | -0.0 ± 0.0 | 86.8 ± 0.7 | 94.3 ± 0.3 |                    19.7 |
| celeba | bottleneck_fixed |       4 | 95.2 ± 0.1 | 40.0 ± 7.3 | 94.5 ± 0.9 | 95.8 ± 0.7 | -0.0 ± 0.0 | 89.4 ± 1.5 | 95.2 ± 0.1 |                    40.8 |
| celeba | mlp              |       4 | 95.2 ± 0.1 | 40.0 ± 4.0 | 94.4 ± 0.1 | 96.1 ± 0.2 | -0.0 ± 0.0 | 90.0 ± 0.4 | 95.2 ± 0.1 |                    39.9 |



Paired worst-group difference vs MLP (avg selection):

| task   | comparison             |   seeds | ΔWGA (pp)   |
|:-------|:-----------------------|--------:|:------------|
| celeba | bottleneck − mlp       |       4 | -19.2 ± 8.3 |
| celeba | bottleneck_fixed − mlp |       3 | +0.4 ± 7.7  |

