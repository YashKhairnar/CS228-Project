# AVConsist — Cross-Modal Audio-Visual Deepfake Detection
**Yash Khairnar | San Jose State University | Checkpoint 2**

---

## Project Summary

AVConsist is a cross-modal attention-based deepfake detection framework that jointly
encodes face frames (ViT-B/16) and voice (Audio Spectrogram Transformer) and uses
bidirectional cross-modal attention to detect audio-visual inconsistencies introduced
by deepfake generation.

**Checkpoint 2 Results vs Checkpoint 1 Baseline:**

| Metric   | ResNet-50 Baseline | AVConsist (CP2) | Improvement |
|----------|--------------------|-----------------|-------------|
| Accuracy | 80.07%             | 91.20%          | +11.13%     |
| FAR      | 15.79%             | 5.80%           | −9.99%      |
| FRR      | 24.03%             | 9.40%           | −14.63%     |
| EER      | 18.95%             | 6.90%           | −12.05%     |
| AUC      | 0.8736             | 0.9610          | +0.0874     |

---

## Repository Structure

```
avfake_project/
├── README.md
├── requirements.txt
│
├── src/
│   ├── models.py        — FaceEncoder, VoiceEncoder, CrossModalAttention, AVConsist
│   ├── preprocess.py    — Extract face frames + mel spectrograms from FakeAVCeleb
│   ├── train.py         — Training loop with warmup/finetune strategy
│   └── evaluate.py      — FAR, FRR, EER, AUC, ROC plots, ablation study
│
├── notebooks/
│   └── AVConsist_Training_Colab.ipynb  — Complete Google Colab notebook
│
├── results/
│   ├── test_metrics.json      — Final test set metrics
│   ├── ablation.json          — Ablation study results
│   ├── history.json           — Training history (loss, acc, EER per epoch)
│   ├── roc_farfrr.png         — ROC + FAR/FRR plot
│   └── training_curves.png    — Training dynamics plot
│
└── checkpoints/
    └── best_model.pt          — Saved weights (best validation EER)
```

---

## Quick Start — Google Colab (Recommended)

1. Upload `AVConsist_Training_Colab.ipynb` to Google Colab
2. Set Runtime → T4 GPU
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Update `FAKEAVCELEB_ROOT` path in Cell 2 to point to your FakeAVCeleb folder
5. Run all cells in order

---

## Local Setup

```bash
# Clone repo
git clone https://github.com/yashkhairnar/avfake-detection
cd avfake-detection

# Install dependencies
pip install -r requirements.txt
sudo apt-get install ffmpeg   # Linux/Mac

# Step 1: Preprocess FakeAVCeleb (run once, ~3-4 hours for full dataset)
python src/preprocess.py \
    --data_root /path/to/FakeAVCeleb \
    --cache_dir ./cache \
    --max_per_folder 500   # remove for full dataset

# Step 2: Train
python src/train.py \
    --data_dir ./cache \
    --out_dir  ./results \
    --epochs   20 \
    --warmup   5

# Step 3: Evaluate
python src/evaluate.py \
    --checkpoint ./results/best_model.pt \
    --data_dir   ./cache \
    --out_dir    ./results \
    --ablation
```

---

## FakeAVCeleb Dataset

Request access at: https://sites.google.com/view/fakeavcelebdash-lab/download

Expected folder structure after download:
```
FakeAVCeleb/
├── RealVideo-RealAudio/    ← label 0 (real)
├── FakeVideo-RealAudio/    ← label 1 (face swap)
├── RealVideo-FakeAudio/    ← label 1 (voice clone)
└── FakeVideo-FakeAudio/    ← label 1 (both faked)
```

---

## Model Architecture

```
Video clip (.mp4)
    ├── Face frames (16 × 224×224) ──► ViT-B/16 ──► F_v (B, 16, 512)
    └── Log-mel spectrogram ──────────► AST ───────► F_a (B, 512)
                                                          │
                                          CrossModalAttention
                                          (bidirectional, 8 heads)
                                                          │
                                          Fusion: concat + mean pool
                                                          │
                                          Classifier (1024→256→1)
                                                          │
                                               REAL (0) / FAKE (1)
```

---

## Key Hyperparameters

| Parameter        | Value     |
|------------------|-----------|
| Face encoder     | ViT-B/16  |
| Voice encoder    | AST       |
| Feature dim      | 512       |
| Attention heads  | 8         |
| Batch size       | 8         |
| Warmup LR        | 1e-4      |
| Fine-tune LR     | 1e-5      |
| Total epochs     | 20        |
| Warmup epochs    | 5         |
| Loss             | Weighted BCE |
| Optimizer        | AdamW     |
| Grad clip        | 1.0       |
