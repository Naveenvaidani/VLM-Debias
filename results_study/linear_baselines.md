# Linear baselines on frozen SigLIP features (mean-pooled), test %, mean ± s.d. over seeds

|                                  |   ('worst_group_acc', 'mean') |   ('worst_group_acc', 'std') |   ('avg_acc', 'mean') |   ('avg_acc', 'std') |
|:---------------------------------|------------------------------:|-----------------------------:|----------------------:|---------------------:|
| ('waterbirds', 'dfr')            |                          84.2 |                          1.5 |                  87.7 |                  1.2 |
| ('waterbirds', 'erm_probe')      |                          62   |                          0   |                  84.9 |                  0   |
| ('waterbirds', 'oracle_gb')      |                          84.3 |                          3   |                  86.4 |                  1.5 |
| ('watermark_p0.95', 'dfr')       |                          85.1 |                          1.3 |                  87.9 |                  0.7 |
| ('watermark_p0.95', 'erm_probe') |                          46.5 |                          2.5 |                  77.4 |                  0.2 |
| ('watermark_p0.95', 'oracle_gb') |                          84.7 |                          1.4 |                  87.4 |                  1   |