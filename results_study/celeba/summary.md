# Connector study results


## Test, selected by validation worst-group accuracy

| task   | connector        |   seeds | Avg        | WGA         | ID          | OOD        | RSR        | Bal.       | Gen.       |   Held-out template WGA |
|:-------|:-----------------|--------:|:-----------|:------------|:------------|:-----------|:-----------|:-----------|:-----------|------------------------:|
| celeba | bottleneck       |       4 | 87.4 ± 9.8 | 36.3 ± 18.7 | 84.4 ± 15.4 | 90.3 ± 6.1 | -0.1 ± 0.2 | 82.6 ± 7.7 | 87.4 ± 9.8 |                    27.8 |
| celeba | bottleneck_fixed |       2 | 94.8 ± 0.3 | 49.3 ± 4.5  | 92.9 ± 0.6  | 96.6 ± 0.0 | -0.0 ± 0.0 | 91.3 ± 0.1 | 94.8 ± 0.3 |                    50.6 |
| celeba | mlp              |       4 | 94.1 ± 1.4 | 54.3 ± 13.7 | 91.5 ± 3.5  | 96.6 ± 0.7 | -0.1 ± 0.0 | 91.6 ± 1.7 | 94.1 ± 1.4 |                    54.3 |



Paired worst-group difference vs MLP (wga selection):

| task   | comparison             |   seeds | ΔWGA (pp)    |
|:-------|:-----------------------|--------:|:-------------|
| celeba | bottleneck − mlp       |       4 | -18.0 ± 28.3 |
| celeba | bottleneck_fixed − mlp |       2 | -7.1 ± 10.3  |



## Test, selected by validation average accuracy (no group labels)

| task   | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR        | Bal.       | Gen.       |   Held-out template WGA |
|:-------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|------------------------:|
| celeba | bottleneck       |       4 | 94.3 ± 0.3 | 20.9 ± 4.3 | 93.6 ± 0.8 | 95.0 ± 0.4 | -0.0 ± 0.0 | 86.8 ± 0.7 | 94.3 ± 0.3 |                    19.7 |
| celeba | bottleneck_fixed |       2 | 95.1 ± 0.1 | 45.7 ± 0.5 | 94.1 ± 1.1 | 96.0 ± 0.9 | -0.0 ± 0.0 | 90.0 ± 1.6 | 95.1 ± 0.1 |                    46.4 |
| celeba | mlp              |       4 | 95.2 ± 0.1 | 40.0 ± 4.0 | 94.4 ± 0.1 | 96.1 ± 0.2 | -0.0 ± 0.0 | 90.0 ± 0.4 | 95.2 ± 0.1 |                    39.9 |



Paired worst-group difference vs MLP (avg selection):

| task   | comparison             |   seeds | ΔWGA (pp)   |
|:-------|:-----------------------|--------:|:------------|
| celeba | bottleneck − mlp       |       4 | -19.2 ± 8.3 |
| celeba | bottleneck_fixed − mlp |       2 | +4.2 ± 5.6  |

