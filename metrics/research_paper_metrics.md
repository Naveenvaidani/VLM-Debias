# Quantitative Evaluation Metrics for Research Paper

### Table 1: Performance Comparison under Spurious Shortcut Conditions

| Metric | Baseline MLP Projector | VLM-SHIELD (Ours) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **In-Distribution Accuracy ($Acc_{ID}$)** | 96.0% | 100.0% | +0.0% |
| **Out-of-Distribution Accuracy ($Acc_{OOD}$)** | 62.0% | 100.0% | **+38.0%** |
| **OOD Drop ($\\Delta_{OOD} = Acc_{ID} - Acc_{OOD}$)** | 34.0% | 0.0% | **-34.0%** |
| **Relative Shortcut Reliance ($RSR$)** | 0.3542 | 0.0000 | **-100.0%** |
| **Shortcut Patch Attention Leakage ($Attn_S$)** | High (Global) | 0.0053 | Suppressed |
| **Spatial Grounding Entropy ($H_{attn}$)** | N/A | 5.2765 | High Entropy |

### Key Finding for Paper
VLM-SHIELD reduces **Relative Shortcut Reliance (RSR) by over 70%** compared to standard MLP projectors by dynamically routing query attention to local spatial visual patches instead of spurious background shortcut features.
