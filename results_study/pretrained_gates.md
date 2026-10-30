# Gate values of released Flamingo-style models (|tanh(alpha)|, mean over each tensor)

| Model | Gate | Layers | Mean | Min | Max | Layers with < 0.05 |
|---|---|---|---|---|---|---|
| HuggingFaceM4/idefics-9b | alpha_cross_attn | 8 | 0.114 | 0.006 | 0.239 | 1 |
| HuggingFaceM4/idefics-9b | alpha_dense | 8 | 0.289 | 0.007 | 0.400 | 1 |
| HuggingFaceM4/idefics-9b-instruct | alpha_cross_attn | 8 | 0.115 | 0.005 | 0.242 | 1 |
| HuggingFaceM4/idefics-9b-instruct | alpha_dense | 8 | 0.292 | 0.003 | 0.408 | 1 |
| HuggingFaceM4/idefics-80b | alpha_cross_attn | 20 | 0.068 | 0.000 | 0.170 | 9 |
| HuggingFaceM4/idefics-80b | alpha_dense | 20 | 0.257 | 0.000 | 0.451 | 2 |
| HuggingFaceM4/idefics-80b-instruct | alpha_cross_attn | 20 | 0.067 | 0.000 | 0.169 | 9 |
| HuggingFaceM4/idefics-80b-instruct | alpha_dense | 20 | 0.257 | 0.000 | 0.451 | 2 |
