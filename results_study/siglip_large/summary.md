# Connector study results


## Test, selected by validation worst-group accuracy

| task            | connector        |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck       |       5 | 89.1 ± 4.2 | 74.5 ± 7.4  | 97.4 ± 0.9 | 80.8 ± 7.6  | 0.2 ± 0.1 | 87.1 ± 3.5 | 89.1 ± 4.2 |                    74.9 |
| waterbirds      | bottleneck_fixed |       4 | 91.4 ± 2.1 | 80.9 ± 2.6  | 98.8 ± 0.1 | 83.9 ± 4.2  | 0.2 ± 0.0 | 90.7 ± 1.6 | 91.4 ± 2.1 |                    80.8 |
| waterbirds      | mlp              |       5 | 92.4 ± 0.6 | 84.1 ± 2.7  | 99.0 ± 0.4 | 85.9 ± 1.5  | 0.1 ± 0.0 | 92.0 ± 0.8 | 92.4 ± 0.6 |                    84.1 |
| waterbirds      | qformer          |       5 | 90.8 ± 1.1 | 80.4 ± 1.8  | 98.6 ± 0.3 | 83.0 ± 2.0  | 0.2 ± 0.0 | 89.8 ± 1.0 | 90.8 ± 1.1 |                    80.3 |
| watermark_p0.95 | bottleneck       |       2 | 84.3 ± 1.2 | 68.8 ± 3.5  | 97.7 ± 0.3 | 70.9 ± 2.7  | 0.3 ± 0.0 | 84.3 ± 1.2 | 84.3 ± 1.2 |                    68.8 |
| watermark_p0.95 | bottleneck_fixed |       2 | 88.8 ± 5.0 | 68.2 ± 17.9 | 97.3 ± 2.0 | 80.2 ± 12.1 | 0.2 ± 0.1 | 88.8 ± 5.0 | 88.8 ± 5.0 |                    68.5 |



Paired worst-group difference vs MLP (wga selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck − mlp       |       5 | -9.6 ± 8.6  |
| waterbirds | bottleneck_fixed − mlp |       4 | -2.9 ± 0.4  |
| waterbirds | qformer − mlp          |       5 | -3.7 ± 3.5  |



## Test, selected by validation average accuracy (no group labels)

| task            | connector        |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck       |       5 | 90.1 ± 4.6 | 72.2 ± 6.0  | 97.4 ± 0.9 | 82.8 ± 8.3  | 0.2 ± 0.1 | 87.3 ± 3.6 | 90.1 ± 4.6 |                    72.9 |
| waterbirds      | bottleneck_fixed |       4 | 93.5 ± 1.0 | 81.3 ± 4.0  | 98.6 ± 0.4 | 88.4 ± 2.3  | 0.1 ± 0.0 | 91.6 ± 1.2 | 93.5 ± 1.0 |                    81.3 |
| waterbirds      | mlp              |       5 | 94.6 ± 0.6 | 81.5 ± 4.0  | 98.9 ± 0.5 | 90.4 ± 1.7  | 0.1 ± 0.0 | 92.5 ± 0.8 | 94.6 ± 0.6 |                    81.4 |
| waterbirds      | qformer          |       5 | 92.1 ± 2.0 | 75.2 ± 5.9  | 98.2 ± 0.3 | 85.9 ± 4.2  | 0.1 ± 0.0 | 89.2 ± 0.9 | 92.1 ± 2.0 |                    75.2 |
| watermark_p0.95 | bottleneck       |       2 | 84.4 ± 1.1 | 61.8 ± 13.5 | 98.0 ± 0.7 | 70.8 ± 2.9  | 0.3 ± 0.0 | 84.4 ± 1.1 | 84.4 ± 1.1 |                    61.8 |
| watermark_p0.95 | bottleneck_fixed |       2 | 88.8 ± 4.9 | 67.9 ± 18.3 | 97.4 ± 2.0 | 80.3 ± 11.9 | 0.2 ± 0.1 | 88.8 ± 4.9 | 88.8 ± 4.9 |                    67.7 |



Paired worst-group difference vs MLP (avg selection):

| task       | comparison             |   seeds | ΔWGA (pp)   |
|:-----------|:-----------------------|--------:|:------------|
| waterbirds | bottleneck − mlp       |       5 | -9.3 ± 7.8  |
| waterbirds | bottleneck_fixed − mlp |       4 | -0.1 ± 2.1  |
| waterbirds | qformer − mlp          |       5 | -6.2 ± 4.9  |

