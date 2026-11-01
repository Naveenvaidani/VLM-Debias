# Connector study results


## Test, selected by validation worst-group accuracy

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       1 | 85.7 ± 0.0 | 72.1 ± 0.0 | 98.3 ± 0.0 | 73.2 ± 0.0 | 0.3 ± 0.0 | 85.9 ± 0.0 | 85.7 ± 0.0 |                    72.7 |
| waterbirds | qformer          |       1 | 87.7 ± 0.0 | 69.8 ± 0.0 | 98.3 ± 0.0 | 77.0 ± 0.0 | 0.2 ± 0.0 | 85.7 ± 0.0 | 87.7 ± 0.0 |                    69.5 |



## Test, selected by validation average accuracy (no group labels)

| task       | connector        |   seeds | Avg        | WGA        | ID         | OOD        | RSR       | Bal.       | Gen.       |   Held-out template WGA |
|:-----------|:-----------------|--------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|------------------------:|
| waterbirds | bottleneck_fixed |       1 | 90.3 ± 0.0 | 64.7 ± 0.0 | 98.1 ± 0.0 | 82.6 ± 0.0 | 0.2 ± 0.0 | 86.1 ± 0.0 | 90.3 ± 0.0 |                    64.5 |
| waterbirds | qformer          |       1 | 91.2 ± 0.0 | 72.1 ± 0.0 | 98.1 ± 0.0 | 84.2 ± 0.0 | 0.1 ± 0.0 | 88.1 ± 0.0 | 91.2 ± 0.0 |                    72.1 |

