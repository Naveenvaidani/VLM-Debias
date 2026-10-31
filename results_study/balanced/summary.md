# Connector study results


## Test, selected by validation worst-group accuracy

| task            | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| watermark_p0.95 | bottleneck_fixed |       3 | 91.4 ± 1.0 | 87.1 ± 3.5 | 94.9 ± 1.6 | 88.0 ± 3.7 | 0.1 ± 0.1 | 91.4 ± 1.0 | 91.4 ± 1.0 |                    86.9 |
| watermark_p0.95 | mlp              |       3 | 92.3 ± 0.4 | 89.1 ± 0.9 | 94.1 ± 0.9 | 90.6 ± 1.7 | 0.0 ± 0.0 | 92.3 ± 0.4 | 92.3 ± 0.4 |                    89.2 |



Paired worst-group difference vs MLP (wga selection):

| task            | comparison             |   seeds | ΔWGA (pp)   |
|:----------------|:-----------------------|--------:|:------------|
| watermark_p0.95 | bottleneck_fixed − mlp |       3 | -2.1 ± 4.1  |



## Test, selected by validation average accuracy (no group labels)

| task            | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| watermark_p0.95 | bottleneck_fixed |       3 | 91.4 ± 1.0 | 87.1 ± 3.5 | 94.9 ± 1.6 | 88.0 ± 3.7 | 0.1 ± 0.1 | 91.4 ± 1.0 | 91.4 ± 1.0 |                    86.9 |
| watermark_p0.95 | mlp              |       3 | 92.3 ± 0.4 | 89.1 ± 0.9 | 94.1 ± 0.9 | 90.6 ± 1.7 | 0.0 ± 0.0 | 92.3 ± 0.4 | 92.3 ± 0.4 |                    89.2 |



Paired worst-group difference vs MLP (avg selection):

| task            | comparison             |   seeds | ΔWGA (pp)   |
|:----------------|:-----------------------|--------:|:------------|
| watermark_p0.95 | bottleneck_fixed − mlp |       3 | -2.1 ± 4.1  |

