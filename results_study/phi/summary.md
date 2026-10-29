# Connector study results


## Test, selected by validation worst-group accuracy

| task            | connector        |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck       |       3 | 82.9 ± 1.4 | 61.0 ± 7.3  | 97.6 ± 0.2 | 68.3 ± 2.7  | 0.3 ± 0.0 | 80.6 ± 1.8 | 82.9 ± 1.4 |                    60.4 |
| waterbirds      | bottleneck_fixed |       3 | 86.2 ± 1.7 | 69.8 ± 5.1  | 98.3 ± 0.1 | 74.1 ± 3.5  | 0.2 ± 0.0 | 84.9 ± 1.8 | 86.2 ± 1.7 |                    69.6 |
| waterbirds      | mlp              |       3 | 91.4 ± 1.5 | 77.1 ± 1.7  | 99.0 ± 0.1 | 83.8 ± 2.9  | 0.2 ± 0.0 | 89.7 ± 0.8 | 91.4 ± 1.5 |                    76.7 |
| waterbirds      | qformer          |       3 | 87.0 ± 2.6 | 73.2 ± 3.4  | 98.3 ± 0.1 | 75.6 ± 5.1  | 0.2 ± 0.1 | 86.7 ± 1.6 | 87.0 ± 2.6 |                    73.4 |
| watermark_p0.95 | bottleneck       |       3 | 75.7 ± 1.8 | 42.3 ± 13.3 | 98.3 ± 0.5 | 53.1 ± 4.1  | 0.5 ± 0.0 | 75.7 ± 1.8 | 75.7 ± 1.7 |                    42.3 |
| watermark_p0.95 | bottleneck_fixed |       3 | 80.2 ± 6.2 | 52.5 ± 11.5 | 97.2 ± 1.4 | 63.2 ± 12.6 | 0.3 ± 0.1 | 80.2 ± 6.2 | 80.2 ± 6.2 |                    52.8 |
| watermark_p0.95 | mlp              |       3 | 86.7 ± 3.0 | 64.1 ± 12.3 | 97.0 ± 2.0 | 76.4 ± 7.7  | 0.2 ± 0.1 | 86.7 ± 3.0 | 86.6 ± 3.0 |                    60.3 |
| watermark_p0.95 | qformer          |       3 | 87.0 ± 4.0 | 67.9 ± 14.7 | 97.1 ± 2.1 | 76.9 ± 10.2 | 0.2 ± 0.1 | 87.0 ± 4.0 | 87.0 ± 4.0 |                    67.7 |



Paired worst-group difference vs MLP (wga selection):

| task            | comparison             |   seeds | ΔWGA (pp)   |
|:----------------|:-----------------------|--------:|:------------|
| waterbirds      | bottleneck − mlp       |       3 | -16.1 ± 7.4 |
| watermark_p0.95 | bottleneck − mlp       |       3 | -21.8 ± 1.4 |
| waterbirds      | bottleneck_fixed − mlp |       3 | -7.3 ± 6.5  |
| watermark_p0.95 | bottleneck_fixed − mlp |       3 | -11.6 ± 7.3 |
| waterbirds      | qformer − mlp          |       3 | -3.9 ± 4.8  |
| watermark_p0.95 | qformer − mlp          |       3 | +3.7 ± 10.0 |



## Test, selected by validation average accuracy (no group labels)

| task            | connector        |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:-----------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck       |       3 | 83.8 ± 2.5 | 59.6 ± 9.3  | 97.5 ± 0.3 | 70.0 ± 5.1  | 0.3 ± 0.1 | 80.9 ± 1.7 | 83.8 ± 2.5 |                    59.3 |
| waterbirds      | bottleneck_fixed |       3 | 88.3 ± 2.1 | 59.8 ± 9.7  | 98.1 ± 0.2 | 78.5 ± 4.2  | 0.2 ± 0.0 | 84.0 ± 2.4 | 88.3 ± 2.1 |                    59.8 |
| waterbirds      | mlp              |       3 | 93.0 ± 1.0 | 74.2 ± 4.3  | 98.4 ± 1.0 | 87.5 ± 2.9  | 0.1 ± 0.0 | 89.7 ± 0.9 | 93.0 ± 1.0 |                    73.4 |
| waterbirds      | qformer          |       3 | 90.4 ± 0.6 | 68.8 ± 9.4  | 98.2 ± 0.2 | 82.7 ± 1.4  | 0.2 ± 0.0 | 87.0 ± 1.6 | 90.4 ± 0.6 |                    68.7 |
| watermark_p0.95 | bottleneck       |       3 | 75.9 ± 1.8 | 34.1 ± 10.7 | 98.5 ± 0.5 | 53.3 ± 4.1  | 0.5 ± 0.0 | 75.9 ± 1.8 | 75.9 ± 1.8 |                    34.3 |
| watermark_p0.95 | bottleneck_fixed |       3 | 82.6 ± 2.6 | 51.9 ± 12.2 | 97.8 ± 1.5 | 67.4 ± 6.7  | 0.3 ± 0.1 | 82.6 ± 2.6 | 82.6 ± 2.6 |                    52.2 |
| watermark_p0.95 | mlp              |       3 | 86.7 ± 3.0 | 64.1 ± 12.3 | 97.0 ± 2.0 | 76.4 ± 7.7  | 0.2 ± 0.1 | 86.7 ± 3.0 | 86.6 ± 3.0 |                    60.3 |
| watermark_p0.95 | qformer          |       3 | 87.0 ± 4.0 | 67.9 ± 14.7 | 97.1 ± 2.1 | 76.9 ± 10.2 | 0.2 ± 0.1 | 87.0 ± 4.0 | 87.0 ± 4.0 |                    67.7 |



Paired worst-group difference vs MLP (avg selection):

| task            | comparison             |   seeds | ΔWGA (pp)    |
|:----------------|:-----------------------|--------:|:-------------|
| waterbirds      | bottleneck − mlp       |       3 | -14.6 ± 5.2  |
| watermark_p0.95 | bottleneck − mlp       |       3 | -30.0 ± 12.8 |
| waterbirds      | bottleneck_fixed − mlp |       3 | -14.4 ± 5.4  |
| watermark_p0.95 | bottleneck_fixed − mlp |       3 | -12.2 ± 6.8  |
| waterbirds      | qformer − mlp          |       3 | -5.4 ± 12.2  |
| watermark_p0.95 | qformer − mlp          |       3 | +3.7 ± 10.0  |

