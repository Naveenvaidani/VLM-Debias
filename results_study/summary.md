# Connector study results


## Test, selected by validation worst-group accuracy

| task            | connector   |   seeds | Avg        | WGA         | ID         | OOD         | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:------------|--------:|:-----------|:------------|:-----------|:------------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck  |       3 | 83.0 ± 8.6 | 62.1 ± 14.9 | 96.6 ± 2.1 | 69.4 ± 15.0 | 0.3 ± 0.1 | 80.9 ± 8.4 | 83.0 ± 8.6 |                    63.9 |
| waterbirds      | mlp         |       3 | 90.4 ± 1.5 | 74.5 ± 1.5  | 98.9 ± 0.2 | 81.9 ± 3.1  | 0.2 ± 0.0 | 88.6 ± 1.0 | 90.4 ± 1.5 |                    74.4 |
| waterbirds      | qformer     |       3 | 87.6 ± 1.8 | 71.7 ± 2.5  | 98.1 ± 0.2 | 77.1 ± 3.6  | 0.2 ± 0.0 | 86.2 ± 0.6 | 87.6 ± 1.8 |                    71.9 |
| watermark_p0.95 | bottleneck  |       3 | 79.7 ± 4.1 | 50.3 ± 16.1 | 98.2 ± 0.7 | 61.2 ± 8.3  | 0.4 ± 0.1 | 79.7 ± 4.1 | 79.7 ± 4.1 |                    48.7 |
| watermark_p0.95 | mlp         |       3 | 84.9 ± 0.4 | 63.6 ± 4.4  | 98.6 ± 0.3 | 71.2 ± 0.6  | 0.3 ± 0.0 | 84.9 ± 0.4 | 84.9 ± 0.4 |                    64   |
| watermark_p0.95 | qformer     |       3 | 87.3 ± 1.3 | 68.6 ± 4.1  | 98.1 ± 0.5 | 76.5 ± 3.1  | 0.2 ± 0.0 | 87.3 ± 1.3 | 87.3 ± 1.3 |                    68.4 |



Paired worst-group difference vs MLP (wga selection):

| task            | comparison       |   seeds | ΔWGA (pp)    |
|:----------------|:-----------------|--------:|:-------------|
| waterbirds      | qformer − mlp    |       3 | -2.8 ± 1.7   |
| watermark_p0.95 | qformer − mlp    |       3 | +5.0 ± 8.5   |
| waterbirds      | bottleneck − mlp |       3 | -12.4 ± 14.4 |
| watermark_p0.95 | bottleneck − mlp |       3 | -13.3 ± 17.6 |



## Test, selected by validation average accuracy (no group labels)

| task            | connector   |   seeds | Avg        | WGA         | ID         | OOD        | RSR       | Bal.        | Gen.       |   Held-out template WGA |
|:----------------|:------------|--------:|:-----------|:------------|:-----------|:-----------|:----------|:------------|:-----------|------------------------:|
| waterbirds      | bottleneck  |       3 | 87.5 ± 3.2 | 48.5 ± 34.2 | 96.5 ± 2.2 | 78.4 ± 4.5 | 0.2 ± 0.0 | 80.2 ± 10.0 | 87.5 ± 3.2 |                    48.6 |
| waterbirds      | mlp         |       3 | 91.7 ± 1.1 | 71.7 ± 2.3  | 98.3 ± 0.6 | 85.2 ± 2.8 | 0.1 ± 0.0 | 88.4 ± 0.7  | 91.7 ± 1.1 |                    71.5 |
| waterbirds      | qformer     |       3 | 89.0 ± 0.8 | 69.3 ± 2.9  | 98.2 ± 0.0 | 79.8 ± 1.5 | 0.2 ± 0.0 | 86.3 ± 1.0  | 89.0 ± 0.8 |                    69.3 |
| watermark_p0.95 | bottleneck  |       3 | 79.7 ± 4.1 | 49.7 ± 17.2 | 98.2 ± 0.7 | 61.2 ± 8.3 | 0.4 ± 0.1 | 79.7 ± 4.1  | 79.7 ± 4.1 |                    48.2 |
| watermark_p0.95 | mlp         |       3 | 85.8 ± 1.2 | 62.7 ± 3.7  | 97.7 ± 1.9 | 74.0 ± 4.2 | 0.2 ± 0.1 | 85.8 ± 1.2  | 85.8 ± 1.2 |                    62.9 |
| watermark_p0.95 | qformer     |       3 | 86.9 ± 1.8 | 66.8 ± 6.8  | 97.6 ± 0.6 | 76.2 ± 3.5 | 0.2 ± 0.0 | 86.9 ± 1.8  | 86.9 ± 1.8 |                    66.3 |



Paired worst-group difference vs MLP (avg selection):

| task            | comparison       |   seeds | ΔWGA (pp)    |
|:----------------|:-----------------|--------:|:-------------|
| waterbirds      | qformer − mlp    |       3 | -2.4 ± 1.0   |
| watermark_p0.95 | qformer − mlp    |       3 | +4.1 ± 10.1  |
| waterbirds      | bottleneck − mlp |       3 | -23.1 ± 34.8 |
| watermark_p0.95 | bottleneck − mlp |       3 | -13.0 ± 18.2 |

