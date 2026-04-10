# Quick Results (fixed split 7/1/2, seed=42)

| Model | Acc | F1_w | F1_m |
|---|---:|---:|---:|
| baseline_mfcc | 0.9043 | 0.9047 | 0.8880 |
| model_b_no_cr | 0.9246 | 0.9245 | 0.9059 |
| model_b_cr | 0.9336 | 0.9337 | 0.9110 |
| model_b_chroma_no_cr | 0.9260 | 0.9260 | 0.9159 |
| model_b_chroma_cr | 0.9424 | 0.9424 | 0.9245 |
| model_b_pcen_no_cr | 0.9242 | 0.9241 | 0.9088 |
| model_b_pcen_cr | 0.9410 | 0.9411 | 0.9113 |
| model_b_spectral_no_cr | 0.9270 | 0.9268 | 0.9052 |
| model_b_spectral_cr | 0.9385 | 0.9384 | 0.9270 |

## Takeaways
- CR 在 PCEN/Spectral 等新增特征下依然稳定提升
- 当前 quick 最优：model_b_chroma_cr（Acc=0.9424）
