# Audio-Visual Deepfake Detection

This project provides a complete pipeline for training and evaluating an **Audio-Visual Consistency (AVConsist)** model to detect deepfakes. By analyzing both the visual (facial) and audio streams of a video, the model learns to identify manipulated content by detecting inconsistencies across these two modalities.

The project uses the [**Localized Audio-Visual DeepFake (LAV-DF)**](https://www.kaggle.com/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df) dataset from Kaggle. The provided script will automatically download the dataset via the Kaggle API, extract features, train the model, and generate evaluation metrics.

## Architecture

The core model, `AVConsist`, consists of three main components:
1. **Face Encoder**: A Vision Transformer (ViT) that extracts rich spatial features from video frames.
2. **Voice Encoder**: A CNN-based architecture that processes Mel-spectrograms generated from the audio track.
3. **Cross-Modal Attention**: A multi-head attention module that fuses the facial and vocal embeddings. It cross-references audio features with visual features (and vice-versa) before passing the unified representation to a final classifier.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need to install the Python dependencies as well as `ffmpeg` for audio extraction.

### Install System Dependencies

You need `ffmpeg` to extract audio from `.mp4` video files.
- **Ubuntu/Debian**: `sudo apt-get install -y ffmpeg`
- **macOS**: `brew install ffmpeg`

### Install Python Dependencies

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## ⚙️ Usage

To run the complete pipeline (downloading, caching, training, and evaluation), simply execute the training script:

```bash
python train_avfake.py
```

### Optional Arguments

You can customize the directories used for caching features and saving results:

```bash
python train_avfake.py --cache-dir ./my_cache --save-dir ./my_results --model-path ./my_results/best_model.pt
```

## 📊 Results & Outputs

The script will train the model over 20 epochs (with an initial warmup phase where the ViT encoder is frozen). After training, it evaluates the best-performing model on a holdout test set and saves the following artifacts to the `avfake_results` directory:

- `best_model.pt`: The PyTorch model weights achieving the best Equal Error Rate (EER) on the validation set.
- `history.json`: Epoch-by-epoch training and validation loss/accuracy/EER logs.
- `test_metrics.json`: Final test metrics including Accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR), EER, and AUC.
- `roc_farfrr.png`: Visualization curves of the ROC and the FAR/FRR trade-off thresholds.

Please reach out via email to `[yashkvk7@gmail.com]` to obtain the download links for the weights.
