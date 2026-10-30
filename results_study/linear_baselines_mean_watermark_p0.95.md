# Linear baselines on frozen SigLIP features (mean-pooled), test %, mean ± s.d. over seeds

|                                          |   ('worst_group_acc', 'mean') |   ('worst_group_acc', 'std') |   ('avg_acc', 'mean') |   ('avg_acc', 'std') |
|:-----------------------------------------|------------------------------:|-----------------------------:|----------------------:|---------------------:|
| ('watermark_p0.95', 'mean', 'dfr')       |                          85.1 |                          1.3 |                  87.9 |                  0.7 |
| ('watermark_p0.95', 'mean', 'erm_probe') |                          46.5 |                          2.5 |                  77.4 |                  0.2 |
| ('watermark_p0.95', 'mean', 'oracle_gb') |                          84.7 |                          1.4 |                  87.4 |                  1   |