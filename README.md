# Deepfake Detection for Biometric Security

This repository contains research and implementations for detecting deepfake media, designed to improve biometric security. The project progressively builds from single-modality visual baseline models to advanced multi-modal (audio-visual) architectures.

The work is organized into checkpoints, each focusing on different datasets, modalities, and deep learning techniques.

---

## 📂 Project Structure

### [Checkpoint 1: Visual Baselines & EDA](./checkpoint1)
**Focus:** Static Image Analysis, Handcrafted Features, and Transfer Learning
**Dataset:** Deepfake Detection Challenge (DFDC) subset

This checkpoint establishes the baseline for the project by analyzing individual video frames. 
- **Exploratory Data Analysis (EDA):** Extracts and visualizes handcrafted features like texture variance, noise residuals, and block artifacts to distinguish between real and fake images.
- **Baseline Model:** Implements a Convolutional Neural Network (CNN) baseline using a pre-trained **ResNet50** model. It fine-tunes the network to classify static faces as real or fake, outputting comprehensive metrics and ROC/FAR/FRR trade-off curves.

**Key File:** `checkpoint1/run_pipeline.py`

### [Checkpoint 2: Audio-Visual Consistency (AVConsist)](./checkpoint2)
**Focus:** Cross-Modal Attention, Multi-modality (Audio & Video)
**Dataset:** Localized Audio-Visual DeepFake (LAV-DF)

Modern deepfakes often have subtle desynchronization or inconsistencies between the audio track and the visual lip/facial movements. This checkpoint addresses these flaws using a multi-modal approach.
- **Face Encoder:** Uses a Vision Transformer (ViT) to extract spatial facial features.
- **Voice Encoder:** Uses a custom CNN on Mel-spectrograms to extract audio features.
- **Cross-Modal Attention:** A multi-head attention mechanism that cross-references the facial and vocal embeddings before passing them to a final classifier, highly improving detection robustness.

**Key File:** `checkpoint2/train_avfake.py`

---

## 🚀 Getting Started

Each checkpoint is fully self-contained with its own scripts, requirements, and execution pipelines. 

To run the models or reproduce the results, please navigate into the respective checkpoint directory and follow the specific instructions in its local `README.md` (if available) or simply inspect the main pipeline script.

```bash
# Example: Running the AVConsist Pipeline
cd checkpoint2
pip install -r requirements.txt
python train_avfake.py
```

## 🔗 Weights & Resources

Pre-trained model weights for the various architectures (ResNet50 baselines, AVConsist) are available upon request. 

Please reach out via email to `[yashkvk7@gmail.com]` to obtain the download links for the weights or the processed subsets of the data.
