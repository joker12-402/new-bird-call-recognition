# Multi-Domain Feature Fusion + Channel Reweighting (CR) for Bird Sound Classification

> Bird sound classification with a lightweight CNN backbone, multi-feature fusion (MFCC/Mel/Chroma/PCEN/Spectral Contrast) and input-channel reweighting (CR).
>
> This repository is organized for **experiment reproducibility**: clear model variants, consistent training protocol, and result summaries.

---

## TL;DR（30秒了解我做了什么）
- **任务**：鸟类声音分类（20类，约 14k 条音频切片）
- **特征**：MFCC、Mel（能量谱）、Chroma、PCEN、Spectral Contrast
- **模型**：轻量 CNN（ImprovedBirdNet）+ 输入端通道重标定 **CR（Channel Reweighting）**
- **评测**：Stratified **5-fold（主结论）** + quick 固定划分 **7/1/2（快速筛选）**
- **结论**：CR 在多特征融合下带来稳定收益；时域特征（RMSE）收益有限

---

## Results
### 5-fold Cross Validation（mean ± std，seed=42）
> 完整表格见：`results/kfold_summary.md`

- **baseline_mfcc**：Acc **0.9244 ± 0.0035**
- **model_b（MFCC+Mel）**：Acc **0.9314 ± 0.0057**
- **model_b_cr（MFCC+Mel + CR）**：Acc **0.9396 ± 0.0031**
- **model_c_cr（MFCC+RMSE+Mel + CR）**：Acc **0.9387 ± 0.0049**

结论（5-fold）：
- **CR 稳定提升**：model_b_cr vs model_b，Acc **+0.0082**；F1_macro **+0.0123**
- **时域特征（RMSE）直接拼接收益有限**：model_a 相比 baseline 略降

### Quick（fixed split 7/1/2, seed=42）
> 完整表格见：`results/quick_summary.md`

- best quick：**model_b_chroma_cr Acc=0.9424**
- PCEN + CR：**Acc=0.9410**
- Spectral + CR：**Acc=0.9385**

---

## Repository Structure
- `models/`
  - `baseline.py`：ImprovedBirdNet（无注意力）
  - `attention_net.py`：CR 模块（输入端通道重标定）+ ImprovedBirdNetWithAttention
- `utils/`
  - `audio_features.py`：特征提取（MFCC / Mel / RMSE 等）
  - `dataset.py`：多种 Dataset（单/双/三通道输入）
- `scripts/`
  - `train_kfold.py`：5-fold 训练/评测入口（当前仓库版本）
- `results/`
  - `kfold_summary.md`：5-fold 汇总（mean±std）
  - `quick_summary.md`：quick 汇总（Acc/F1w/F1m）

---

## Data（重要）
**数据集不包含在本仓库中**（版权/合规原因）。本仓库提供：
- 代码结构与训练脚本
- 统一的实验协议（5-fold / quick）
- 结果汇总与对比表（results/）

你需要自行准备：
- `metadata.json`：样本索引与标签（建议字段：`file_path`, `label`）
- `label_mapping.json`：`label_to_name` 映射
- `audio_dir/`：音频文件目录（脚本内可配置）

> 建议在 `data/` 中仅放 placeholder 说明文件，不要上传原始音频。

---

## Environment
建议 Python 3.9+，核心依赖：
- `torch`
- `librosa`
- `numpy`, `scipy`
- `scikit-learn`
- `tqdm`

（后续我会补充 `requirements.txt`，目前可按上述库安装）

---

## How to Run
### 1) 5-fold Training (main)
当前仓库入口脚本：
```bash
python scripts/train_kfold.py
```

你可以在脚本中配置：
- `AUDIO_DIR / METADATA / LABEL_MAPPING / OUTPUT_ROOT`
- `models_to_run`（选择要跑的模型变体）

输出：每个模型一个目录，包含配置与汇总（json/日志）。


### 2) Quick Compare (for fast screening)
入口脚本：
```bash
python scripts/train_quick_compare_models.py ^
  --audio_dir "D:/paper/data/processed_audio" ^
  --metadata "D:/paper/data/features/metadata.json" ^
  --label_mapping "D:/paper/data/features/label_mapping.json" ^
  --out_dir "D:/paper/quick_compare_results" ^
  --seed 42
```

输出：
- 每个模型一个目录：`{model}_quick_seed{seed}/`（含 config/history/result/report）
- 汇总：`quick_all_results.json`（位于 out_dir）


## Model Variants（Ablation）
- `baseline_mfcc`：MFCC（1ch）
- `model_a`：MFCC + RMSE（2ch）
- `model_b`：MFCC + Mel（2ch）
- `model_b_cr`：MFCC + Mel + CR（2ch）
- `model_c_no_cr`：MFCC + RMSE + Mel（3ch）
- `model_c_cr`：MFCC + RMSE + Mel + CR（3ch）

---

## Notes / TODO
- [ ] 将 quick 脚本整理进仓库（`scripts/train_quick_compare_models.py`）
- [ ] 新增特征（Chroma/PCEN/Spectral Contrast）的 **5-fold** 对比实验
- [ ] 补充 `requirements.txt` 与更清晰的数据格式示例（metadata/label_mapping）

---

## Contact
- GitHub: https://github.com/joker12-402
