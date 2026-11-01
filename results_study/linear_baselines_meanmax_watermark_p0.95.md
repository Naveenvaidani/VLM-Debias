# Linear baselines on frozen SigLIP features (mean-pooled), test %, mean ± s.d. over seeds

|                                             |   ('worst_group_acc', 'mean') |   ('worst_group_acc', 'std') |   ('avg_acc', 'mean') |   ('avg_acc', 'std') |
|:--------------------------------------------|------------------------------:|-----------------------------:|----------------------:|---------------------:|
| ('watermark_p0.95', 'meanmax', 'dfr')       |                          85.1 |                          0.9 |                  88.2 |                  0.2 |
| ('watermark_p0.95', 'meanmax', 'erm_probe') |                          44.2 |                          0   |                  76   |                  0   |
| ('watermark_p0.95', 'meanmax', 'oracle_gb') |                          84.8 |                          0.9 |                  87.4 |                  1.3 |