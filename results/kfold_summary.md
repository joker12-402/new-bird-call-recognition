# 5-fold Cross Validation Results (mean ± std)

seed=42, StratifiedKFold (5-fold)

| Model | Features | CR (Channel Reweighting) | Acc (mean±std) | F1_w (mean±std) | F1_m (mean±std) |
|---|---|---:|---:|---:|---:|
| baseline_mfcc | MFCC | No | 0.9244 ± 0.0035 | 0.9243 ± 0.0034 | 0.9097 ± 0.0039 |
| model_a | MFCC + Temporal | No | 0.9225 ± 0.0012 | 0.9224 ± 0.0009 | 0.9060 ± 0.0121 |
| model_b | MFCC + Mel | No | 0.9314 ± 0.0057 | 0.9314 ± 0.0055 | 0.9198 ± 0.0093 |
| model_b_cr | MFCC + Mel | Yes | 0.9396 ± 0.0031 | 0.9396 ± 0.0031 | 0.9321 ± 0.0081 |
| model_b_pcen_no_cr | MFCC + Mel + PCEN | No | 0.9293 ± 0.0061 | 0.9292 ± 0.0059 | 0.9150 ± 0.0083 |
| model_b_pcen_cr | MFCC + Mel + PCEN | Yes | 0.9404 ± 0.0037 | 0.9403 ± 0.0039 | 0.9309 ± 0.0119 |
| model_b_chroma_no_cr | MFCC + Mel + Chroma | No | 0.9269 ± 0.0061 | 0.9267 ± 0.0060 | 0.9106 ± 0.0107 |
| model_b_chroma_cr | MFCC + Mel + Chroma | Yes | 0.9362 ± 0.0086 | 0.9361 ± 0.0086 | 0.9258 ± 0.0096 |
| model_b_spectral_no_cr | MFCC + Mel + Spectral Contrast | No | 0.9270 ± 0.0026 | 0.9269 ± 0.0026 | 0.9113 ± 0.0115 |
| model_b_spectral_cr | MFCC + Mel + Spectral Contrast | Yes | 0.9370 ± 0.0045 | 0.9369 ± 0.0046 | 0.9273 ± 0.0116 |
| model_c_no_cr | MFCC + Temporal + Mel | No | 0.9264 ± 0.0073 | 0.9264 ± 0.0072 | 0.9123 ± 0.0080 |
| model_c_cr | MFCC + Temporal + Mel | Yes | 0.9387 ± 0.0049 | 0.9387 ± 0.0049 | 0.9331 ± 0.0085 |

## 结果解读

- CR 在多特征融合场景下都能带来稳定的性能提升。
- 在 MFCC + Mel 的基础上,加入 CR 模块可以从 model_b 提升到 model_b_cr。 
- 在三通道特征组合中,model_b_pcen_cr 取得了最佳的整体表现。
- 单纯的时域特征(model_a)相比 MFCC 基线提升有限。
