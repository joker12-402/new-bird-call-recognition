# Multi-Domain Feature Fusion + Channel Reweighting (CR) for Bird Sound Classification

> Bird sound classification with a lightweight CNN backbone, multi-feature fusion (MFCC/Mel/Chroma/PCEN/Spectral Contrast) and input-channel reweighting (CR).
>
> This repository is organized for **experiment reproducibility**: clear model variants, consistent training protocol, and result summaries.

---

## TL;DR（30秒了解我做了什么）
- **任务**：鸟类声音分类（20类，约 14k 条音频切片）
- **特征**：MFCC、Mel（能量谱）、Chroma、PCEN、Spectral Contrast、Temporal energy（RMS）
- **模型**：轻量 CNN（ImprovedBirdNet）+ 输入端通道重标定 **CR（Channel Reweighting）**
- **评测**：Stratified **5-fold（主结论）** + quick 固定划分 **7/1/2（快速筛选）**
- **结论**：CR 在多特征融合下带来稳定收益；简单时域能量（RMS）收益有限

---

## Results
### 5-fold Cross Validation（mean ± std，seed=42）
> 完整表格见：`results/kfold_summary.md`

- **baseline_mfcc**：Acc **0.9244 ± 0.0035**
- **model_b（MFCC+Mel）**：Acc **0.9314 ± 0.0057**
- **model_b_cr（MFCC+Mel + CR）**：Acc **0.9396 ± 0.0031**
- **model_c_cr（MFCC+Temporal(RMS)+Mel + CR）**：Acc **0.9387 ± 0.0049**

结论（5-fold）：
- **CR 稳定提升**：model_b_cr vs model_b，Acc **+0.0082**；F1_macro **+0.0123**
- **简单时域能量（RMS）直接拼接收益有限**：model_a 相比 baseline 略降

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
  - `audio_features.py`：特征提取（MFCC / Mel / Temporal energy (RMS) 等）
  - `dataset.py`：多种 Dataset（单/双/三通道输入）
  - `quick_runner.py`：quick 训练/评测核心逻辑（供脚本入口调用）
- `scripts/`
  - `train_kfold.py`：5-fold 训练/评测入口（命令行参数）
  - `train_quick_compare_models.py`：quick 固定划分训练/对比入口
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
- `metadata.json`：样本索引与标签（建议字段：`file_path`, `label`），通过 `--metadata` 指定
- `label_mapping.json`：`label_to_name` 映射，通过 `--label_mapping` 指定
- `audio_dir/`：音频文件目录，通过 `--audio_dir` 指定

> 建议在 `data/` 中仅放 placeholder 说明文件，不要上传原始音频。

**metadata.json 示例**
```json
[
  {"file_path": "0009/111651_1.wav", "label": 0},
  {"file_path": "0017/344259_1.wav", "label": 5}
]
```

**label_mapping.json 示例**
```json
{
  "label_to_name": {
    "0": "class_0",
    "1": "class_1"
  }
}
```

---

## Environment
建议 Python 3.9+，核心依赖：
- `torch`
- `librosa`
- `numpy`, `scipy`
- `scikit-learn`
- `tqdm`

（建议补充 `requirements.txt` 以便一键复现）

---

## How to Run
### 1) 5-fold Training (main)
当前仓库入口脚本：
```bash
python scripts/train_kfold.py ^
  --audio_dir "D:/paper/data/processed_audio" ^
  --metadata "D:/paper/data/features/metadata.json" ^
  --label_mapping "D:/paper/data/features/label_mapping.json" ^
  --out_dir "D:/paper/big_paper_results" ^
  --models "model_b,model_b_cr,model_c_cr" ^
  --seed 42
```

可选模型名（`--models`）：
- `baseline_mfcc`
- `model_a`（MFCC + Temporal(RMS), 2ch）
- `model_b`（MFCC + Mel, 2ch）
- `model_b_cr`（MFCC + Mel + CR, 2ch）
- `model_c_no_cr`（MFCC + Temporal(RMS) + Mel, 3ch）
- `model_c_cr`（MFCC + Temporal(RMS) + Mel + CR, 3ch）
- 或者使用：`--models "all"`

---

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

---

## Model Variants（Ablation）
- `baseline_mfcc`：MFCC（1ch）
- `model_a`：MFCC + Temporal energy（RMS）（2ch）
- `model_b`：MFCC + Mel（2ch）
- `model_b_cr`：MFCC + Mel + CR（2ch）
- `model_c_no_cr`：MFCC + Temporal energy（RMS）+ Mel（3ch）
- `model_c_cr`：MFCC + Temporal energy（RMS）+ Mel + CR（3ch）

---

## Notes / TODO
- [x] Quick 脚本已整理进仓库（`scripts/train_quick_compare_models.py` + `utils/quick_runner.py`）
- [ ] 新增特征（Chroma/PCEN/Spectral Contrast）的 **5-fold** 对比实验
- [ ] 补充 `requirements.txt` 与更清晰的数据格式示例（metadata/label_mapping）

---

## Contact
- GitHub: https://github.com/joker12-402
