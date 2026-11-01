# Connector study results


## Test, selected by validation worst-group accuracy

| task            | connector        |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck       |       5 | 82.7 ± 3.1 | 60.4 ± 5.7  | 97.8 ± 0.3 | 67.5 ± 6.4  | 0.3 ± 0.1 | 80.7 ± 2.2 | 82.7 ± 3.1 |                    59.9 |
| waterbirds      | bottleneck_fixed |       5 | 86.8 ± 1.5 | 70.1 ± 3.6  | 98.3 ± 0.1 | 75.3 ± 3.1  | 0.2 ± 0.0 | 85.3 ± 1.4 | 86.8 ± 1.5 |                    70.1 |
| waterbirds      | mlp              |       5 | 89.9 ± 2.7 | 76.6 ± 3.6  | 98.8 ± 0.4 | 81.1 ± 5.0  | 0.2 ± 0.0 | 89.0 ± 1.6 | 89.9 ± 2.7 |                    76.4 |
| waterbirds      | qformer          |       5 | 87.1 ± 2.6 | 71.9 ± 3.1  | 98.4 ± 0.1 | 75.9 ± 5.2  | 0.2 ± 0.1 | 86.5 ± 1.3 | 87.1 ± 2.6 |                    71.9 |
| watermark_p0.95 | bottleneck       |       5 | 71.8 ± 6.3 | 28.5 ± 21.3 | 94.6 ± 8.6 | 49.0 ± 6.5  | 0.4 ± 0.1 | 71.8 ± 6.3 | 70.4 ± 9.0 |                    28.8 |
| watermark_p0.95 | bottleneck_fixed |       5 | 82.4 ± 6.4 | 60.1 ± 15.1 | 97.0 ± 1.7 | 67.9 ± 13.8 | 0.3 ± 0.2 | 82.4 ± 6.4 | 82.4 ± 6.4 |                    60.4 |
| watermark_p0.95 | mlp              |       5 | 85.2 ± 3.5 | 62.2 ± 10.0 | 97.7 ± 1.8 | 72.7 ± 8.5  | 0.3 ± 0.1 | 85.2 ± 3.5 | 85.2 ± 3.5 |                    57.1 |
| watermark_p0.95 | qformer          |       5 | 86.5 ± 3.0 | 66.1 ± 10.7 | 97.6 ± 1.7 | 75.4 ± 7.7  | 0.2 ± 0.1 | 86.5 ± 3.0 | 86.5 ± 3.0 |                    66.6 |



Paired worst-group difference vs MLP (wga selection):

| task            | comparison             |   seeds | ΔWGA (pp)    |
|:----------------|:-----------------------|--------:|:-------------|
| waterbirds      | bottleneck − mlp       |       5 | -16.2 ± 5.4  |
| watermark_p0.95 | bottleneck − mlp       |       5 | -33.7 ± 16.4 |
| waterbirds      | bottleneck_fixed − mlp |       5 | -6.5 ± 5.6   |
| watermark_p0.95 | bottleneck_fixed − mlp |       5 | -2.1 ± 18.2  |
| waterbirds      | qformer − mlp          |       5 | -4.7 ± 4.7   |
| watermark_p0.95 | qformer − mlp          |       5 | +3.9 ± 7.8   |



## Test, selected by validation average accuracy (no group labels)

| task            | connector        |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck       |       5 | 84.9 ± 2.4 | 52.4 ± 12.1 | 97.5 ± 0.2 | 72.3 ± 4.9  | 0.3 ± 0.1 | 80.1 ± 1.6 | 84.9 ± 2.4 |                    52.1 |
| waterbirds      | bottleneck_fixed |       5 | 88.2 ± 1.5 | 62.5 ± 8.2  | 98.1 ± 0.2 | 78.3 ± 3.0  | 0.2 ± 0.0 | 84.5 ± 2.0 | 88.2 ± 1.5 |                    62.6 |
| waterbirds      | mlp              |       5 | 92.9 ± 0.8 | 74.7 ± 3.2  | 98.4 ± 0.7 | 87.4 ± 2.3  | 0.1 ± 0.0 | 89.8 ± 0.6 | 92.9 ± 0.8 |                    74.1 |
| waterbirds      | qformer          |       5 | 90.5 ± 0.6 | 68.8 ± 6.7  | 98.3 ± 0.2 | 82.6 ± 1.2  | 0.2 ± 0.0 | 87.1 ± 1.1 | 90.5 ± 0.6 |                    68.8 |
| watermark_p0.95 | bottleneck       |       5 | 72.0 ± 6.1 | 22.9 ± 17.6 | 95.6 ± 6.7 | 48.5 ± 7.3  | 0.5 ± 0.1 | 72.0 ± 6.1 | 72.0 ± 6.1 |                    22.8 |
| watermark_p0.95 | bottleneck_fixed |       5 | 83.8 ± 4.5 | 57.7 ± 16.1 | 97.3 ± 1.8 | 70.2 ± 10.7 | 0.3 ± 0.1 | 83.8 ± 4.5 | 83.8 ± 4.5 |                    57.9 |
| watermark_p0.95 | mlp              |       5 | 86.0 ± 2.3 | 63.0 ± 9.3  | 97.7 ± 1.7 | 74.4 ± 6.2  | 0.2 ± 0.1 | 86.0 ± 2.3 | 86.0 ± 2.4 |                    58.8 |
| watermark_p0.95 | qformer          |       5 | 86.6 ± 3.0 | 66.0 ± 10.8 | 97.5 ± 1.6 | 75.7 ± 7.4  | 0.2 ± 0.1 | 86.6 ± 3.0 | 86.6 ± 3.0 |                    68   |



Paired worst-group difference vs MLP (avg selection):

| task            | comparison             |   seeds | ΔWGA (pp)    |
|:----------------|:-----------------------|--------:|:-------------|
| waterbirds      | bottleneck − mlp       |       5 | -22.3 ± 11.6 |
| watermark_p0.95 | bottleneck − mlp       |       5 | -40.1 ± 16.6 |
| waterbirds      | bottleneck_fixed − mlp |       5 | -12.1 ± 6.0  |
| watermark_p0.95 | bottleneck_fixed − mlp |       5 | -5.3 ± 17.5  |
| waterbirds      | qformer − mlp          |       5 | -5.9 ± 8.7   |
| watermark_p0.95 | qformer − mlp          |       5 | +3.0 ± 7.4   |

