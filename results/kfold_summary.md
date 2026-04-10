# 5-fold Cross Validation Results (mean ± std)

seed=42, StratifiedKFold (5-fold)

| Model | Features | CR (Channel Reweighting) | Acc (mean±std) | F1_w (mean±std) | F1_m (mean±std) |
|---|---|---:|---:|---:|---:|
| baseline_mfcc | MFCC | No | 0.9244 ± 0.0035 | 0.9243 ± 0.0034 | 0.9097 ± 0.0039 |
| model_a | MFCC + Time-domain | No | 0.9225 ± 0.0012 | 0.9224 ± 0.0009 | 0.9060 ± 0.0121 |
| model_b | MFCC + Time-Freq | No | 0.9314 ± 0.0057 | 0.9314 ± 0.0055 | 0.9198 ± 0.0093 |
| model_b_cr | MFCC + Time-Freq | Yes | 0.9396 ± 0.0031 | 0.9396 ± 0.0031 | 0.9321 ± 0.0081 |
| model_c_no_cr | MFCC + Time + Time-Freq | No | 0.9264 ± 0.0073 | 0.9264 ± 0.0072 | 0.9123 ± 0.0080 |
| model_c_cr | MFCC + Time + Time-Freq | Yes | 0.9387 ± 0.0049 | 0.9387 ± 0.0049 | 0.9331 ± 0.0085 |

## Notes
- CR 在多特征融合下带来稳定收益：model_b_cr 相比 model_b，Acc +0.0082；F1_m +0.0123
- 时域特征（model_a）整体收益有限：相比 baseline_mfcc 略降
