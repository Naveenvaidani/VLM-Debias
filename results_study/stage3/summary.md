# Connector study results


## Test, selected by validation worst-group accuracy

| task            | connector               |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:------------------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck_fixed        |       5 | 88.4 ± 1.7 | 73.9 ± 2.5 | 98.0 ± 0.3 | 78.9 ± 3.4 | 0.2 ± 0.0 | 86.7 ± 1.1 | 88.4 ± 1.7 |                    73.7 |
| waterbirds      | bottleneck_fixed_dfr    |       5 | 88.7 ± 4.0 | 74.1 ± 5.6 | 97.3 ± 0.7 | 80.1 ± 7.4 | 0.2 ± 0.1 | 88.3 ± 1.4 | 88.7 ± 4.0 |                    74   |
| waterbirds      | bottleneck_fixed_jtt    |       1 | 88.7 ± 0.0 | 68.4 ± 0.0 | 97.8 ± 0.0 | 79.5 ± 0.0 | 0.2 ± 0.0 | 85.8 ± 0.0 | 88.7 ± 0.0 |                    68.2 |
| waterbirds      | bottleneck_fixed_prompt |       1 | 86.8 ± 0.0 | 72.4 ± 0.0 | 98.0 ± 0.0 | 75.5 ± 0.0 | 0.2 ± 0.0 | 85.4 ± 0.0 | 86.8 ± 0.0 |                    72.3 |
| waterbirds      | mlp                     |       5 | 89.5 ± 1.5 | 74.9 ± 1.4 | 98.9 ± 0.2 | 80.2 ± 3.2 | 0.2 ± 0.0 | 88.5 ± 0.3 | 89.5 ± 1.5 |                    75   |
| waterbirds      | mlp_dfr                 |       5 | 89.6 ± 2.8 | 75.0 ± 3.6 | 98.3 ± 0.4 | 80.8 ± 5.3 | 0.2 ± 0.1 | 89.4 ± 0.7 | 89.6 ± 2.8 |                    75.4 |
| waterbirds      | mlp_jtt                 |       1 | 92.1 ± 0.0 | 80.2 ± 0.0 | 98.6 ± 0.0 | 85.6 ± 0.0 | 0.1 ± 0.0 | 90.6 ± 0.0 | 92.1 ± 0.0 |                    80.1 |
| waterbirds      | mlp_prompt              |       1 | 88.5 ± 0.0 | 77.3 ± 0.0 | 98.8 ± 0.0 | 78.1 ± 0.0 | 0.2 ± 0.0 | 88.8 ± 0.0 | 88.5 ± 0.0 |                    77.8 |
| watermark_p0.95 | bottleneck_fixed        |       5 | 87.2 ± 2.7 | 72.5 ± 9.5 | 97.3 ± 0.9 | 77.1 ± 6.3 | 0.2 ± 0.1 | 87.2 ± 2.7 | 87.2 ± 2.7 |                    72.3 |
| watermark_p0.95 | bottleneck_fixed_dfr    |       5 | 92.8 ± 0.6 | 89.6 ± 0.5 | 93.9 ± 0.8 | 91.7 ± 1.7 | 0.0 ± 0.0 | 92.8 ± 0.6 | 92.8 ± 0.6 |                    89.6 |
| watermark_p0.95 | bottleneck_fixed_jtt    |       1 | 82.7 ± 0.0 | 49.9 ± 0.0 | 98.6 ± 0.0 | 66.9 ± 0.0 | 0.3 ± 0.0 | 82.7 ± 0.0 | 82.7 ± 0.0 |                    50.2 |
| watermark_p0.95 | bottleneck_fixed_prompt |       1 | 85.1 ± 0.0 | 65.2 ± 0.0 | 98.3 ± 0.0 | 71.8 ± 0.0 | 0.3 ± 0.0 | 85.1 ± 0.0 | 85.1 ± 0.0 |                    65.3 |
| watermark_p0.95 | mlp                     |       5 | 85.6 ± 2.2 | 64.9 ± 8.9 | 98.5 ± 0.7 | 72.7 ± 4.9 | 0.3 ± 0.1 | 85.6 ± 2.2 | 85.6 ± 2.2 |                    65.7 |
| watermark_p0.95 | mlp_dfr                 |       5 | 90.8 ± 0.9 | 81.2 ± 2.2 | 95.4 ± 1.4 | 86.2 ± 3.2 | 0.1 ± 0.0 | 90.8 ± 0.9 | 90.8 ± 0.9 |                    81.1 |
| watermark_p0.95 | mlp_jtt                 |       1 | 84.3 ± 0.0 | 58.6 ± 0.0 | 99.0 ± 0.0 | 69.6 ± 0.0 | 0.3 ± 0.0 | 84.3 ± 0.0 | 84.3 ± 0.0 |                    58.8 |
| watermark_p0.95 | mlp_prompt              |       1 | 85.1 ± 0.0 | 61.4 ± 0.0 | 99.0 ± 0.0 | 71.1 ± 0.0 | 0.3 ± 0.0 | 85.1 ± 0.0 | 85.1 ± 0.0 |                    61.5 |



Paired worst-group difference vs MLP (wga selection):

| task            | comparison                    |   seeds | ΔWGA (pp)   |
|:----------------|:------------------------------|--------:|:------------|
| waterbirds      | bottleneck_fixed − mlp        |       5 | -1.0 ± 3.3  |
| watermark_p0.95 | bottleneck_fixed − mlp        |       5 | +7.6 ± 3.9  |
| waterbirds      | bottleneck_fixed_dfr − mlp    |       5 | -0.7 ± 5.4  |
| watermark_p0.95 | bottleneck_fixed_dfr − mlp    |       5 | +24.7 ± 9.3 |
| waterbirds      | bottleneck_fixed_jtt − mlp    |       1 | -8.9 ± 0.0  |
| watermark_p0.95 | bottleneck_fixed_jtt − mlp    |       1 | -9.4 ± 0.0  |
| waterbirds      | bottleneck_fixed_prompt − mlp |       1 | -4.9 ± 0.0  |
| watermark_p0.95 | bottleneck_fixed_prompt − mlp |       1 | +5.8 ± 0.0  |
| waterbirds      | mlp_dfr − mlp                 |       5 | +0.2 ± 3.9  |
| watermark_p0.95 | mlp_dfr − mlp                 |       5 | +16.3 ± 7.1 |
| waterbirds      | mlp_jtt − mlp                 |       1 | +2.9 ± 0.0  |
| watermark_p0.95 | mlp_jtt − mlp                 |       1 | -0.8 ± 0.0  |
| waterbirds      | mlp_prompt − mlp              |       1 | -0.0 ± 0.0  |
| watermark_p0.95 | mlp_prompt − mlp              |       1 | +2.1 ± 0.0  |



## Test, selected by validation average accuracy (no group labels)

| task            | connector               |   seeds | Avg        | WGA         | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:----------------|:------------------------|--------:|:-----------|:------------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds      | bottleneck_fixed        |       5 | 89.7 ± 1.4 | 65.5 ± 10.5 | 98.0 ± 0.2 | 81.4 ± 3.0 | 0.2 ± 0.0 | 85.8 ± 2.1 | 89.7 ± 1.4 |                    65   |
| waterbirds      | bottleneck_fixed_dfr    |       5 | 88.7 ± 4.0 | 74.1 ± 5.6  | 97.3 ± 0.7 | 80.1 ± 7.4 | 0.2 ± 0.1 | 88.3 ± 1.4 | 88.7 ± 4.0 |                    74   |
| waterbirds      | bottleneck_fixed_jtt    |       1 | 88.7 ± 0.0 | 68.4 ± 0.0  | 97.8 ± 0.0 | 79.5 ± 0.0 | 0.2 ± 0.0 | 85.8 ± 0.0 | 88.7 ± 0.0 |                    68.2 |
| waterbirds      | bottleneck_fixed_prompt |       1 | 86.8 ± 0.0 | 72.4 ± 0.0  | 98.0 ± 0.0 | 75.5 ± 0.0 | 0.2 ± 0.0 | 85.4 ± 0.0 | 86.8 ± 0.0 |                    72.3 |
| waterbirds      | mlp                     |       5 | 92.0 ± 1.1 | 73.7 ± 3.1  | 98.5 ± 0.4 | 85.5 ± 2.4 | 0.1 ± 0.0 | 89.1 ± 1.0 | 92.0 ± 1.1 |                    73.6 |
| waterbirds      | mlp_dfr                 |       5 | 89.6 ± 2.8 | 75.0 ± 3.6  | 98.3 ± 0.4 | 80.8 ± 5.3 | 0.2 ± 0.1 | 89.4 ± 0.7 | 89.6 ± 2.8 |                    75.4 |
| waterbirds      | mlp_jtt                 |       1 | 92.1 ± 0.0 | 80.2 ± 0.0  | 98.6 ± 0.0 | 85.6 ± 0.0 | 0.1 ± 0.0 | 90.6 ± 0.0 | 92.1 ± 0.0 |                    80.1 |
| waterbirds      | mlp_prompt              |       1 | 88.5 ± 0.0 | 77.3 ± 0.0  | 98.8 ± 0.0 | 78.1 ± 0.0 | 0.2 ± 0.0 | 88.8 ± 0.0 | 88.5 ± 0.0 |                    77.8 |
| watermark_p0.95 | bottleneck_fixed        |       5 | 87.2 ± 2.7 | 72.5 ± 9.5  | 97.3 ± 0.9 | 77.1 ± 6.3 | 0.2 ± 0.1 | 87.2 ± 2.7 | 87.2 ± 2.7 |                    72.3 |
| watermark_p0.95 | bottleneck_fixed_dfr    |       5 | 92.8 ± 0.6 | 89.6 ± 0.5  | 93.9 ± 0.8 | 91.7 ± 1.7 | 0.0 ± 0.0 | 92.8 ± 0.6 | 92.8 ± 0.6 |                    89.6 |
| watermark_p0.95 | bottleneck_fixed_jtt    |       1 | 82.7 ± 0.0 | 49.9 ± 0.0  | 98.6 ± 0.0 | 66.9 ± 0.0 | 0.3 ± 0.0 | 82.7 ± 0.0 | 82.7 ± 0.0 |                    50.2 |
| watermark_p0.95 | bottleneck_fixed_prompt |       1 | 85.1 ± 0.0 | 65.2 ± 0.0  | 98.3 ± 0.0 | 71.8 ± 0.0 | 0.3 ± 0.0 | 85.1 ± 0.0 | 85.1 ± 0.0 |                    65.3 |
| watermark_p0.95 | mlp                     |       5 | 85.6 ± 2.2 | 63.8 ± 8.2  | 98.7 ± 0.6 | 72.4 ± 4.9 | 0.3 ± 0.1 | 85.6 ± 2.2 | 85.6 ± 2.2 |                    64.1 |
| watermark_p0.95 | mlp_dfr                 |       5 | 90.8 ± 0.9 | 81.2 ± 2.2  | 95.4 ± 1.4 | 86.2 ± 3.2 | 0.1 ± 0.0 | 90.8 ± 0.9 | 90.8 ± 0.9 |                    81.1 |
| watermark_p0.95 | mlp_jtt                 |       1 | 84.3 ± 0.0 | 58.6 ± 0.0  | 99.0 ± 0.0 | 69.6 ± 0.0 | 0.3 ± 0.0 | 84.3 ± 0.0 | 84.3 ± 0.0 |                    58.8 |
| watermark_p0.95 | mlp_prompt              |       1 | 85.1 ± 0.0 | 61.4 ± 0.0  | 99.0 ± 0.0 | 71.1 ± 0.0 | 0.3 ± 0.0 | 85.1 ± 0.0 | 85.1 ± 0.0 |                    61.5 |



Paired worst-group difference vs MLP (avg selection):

| task            | comparison                    |   seeds | ΔWGA (pp)   |
|:----------------|:------------------------------|--------:|:------------|
| waterbirds      | bottleneck_fixed − mlp        |       5 | -8.2 ± 10.0 |
| watermark_p0.95 | bottleneck_fixed − mlp        |       5 | +8.7 ± 4.5  |
| waterbirds      | bottleneck_fixed_dfr − mlp    |       5 | +0.5 ± 6.5  |
| watermark_p0.95 | bottleneck_fixed_dfr − mlp    |       5 | +25.8 ± 8.6 |
| waterbirds      | bottleneck_fixed_jtt − mlp    |       1 | -5.9 ± 0.0  |
| watermark_p0.95 | bottleneck_fixed_jtt − mlp    |       1 | -9.4 ± 0.0  |
| waterbirds      | bottleneck_fixed_prompt − mlp |       1 | -1.9 ± 0.0  |
| watermark_p0.95 | bottleneck_fixed_prompt − mlp |       1 | +5.8 ± 0.0  |
| waterbirds      | mlp_dfr − mlp                 |       5 | +1.4 ± 6.2  |
| watermark_p0.95 | mlp_dfr − mlp                 |       5 | +17.4 ± 6.2 |
| waterbirds      | mlp_jtt − mlp                 |       1 | +6.0 ± 0.0  |
| watermark_p0.95 | mlp_jtt − mlp                 |       1 | -0.8 ± 0.0  |
| waterbirds      | mlp_prompt − mlp              |       1 | +3.0 ± 0.0  |
| watermark_p0.95 | mlp_prompt − mlp              |       1 | +2.1 ± 0.0  |

