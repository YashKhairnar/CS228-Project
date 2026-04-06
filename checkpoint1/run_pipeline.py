import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              classification_report, accuracy_score)
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings('ignore')

BASE    = Path(__file__).parent
DATA    = BASE / 'data'
FIGS    = BASE / 'figures'
OUT     = BASE / 'output'
FIGS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ── 1. Load dataset paths ────────────────────────────────────────────────────
def load_dataset():
    records = []
    for label, folder in [(0, 'real'), (1, 'fake')]:
        p = DATA / folder
        for f in sorted(p.glob('*.png')) :
            records.append({'path': str(f), 'label': label})
    df = pd.DataFrame(records)
    print(f"Dataset loaded: {len(df)} samples | Real: {(df.label==0).sum()} | Fake: {(df.label==1).sum()}")
    return df

# ── 2. Image feature extraction ──────────────────────────────────────────────
def extract_features(df, sample_n=None):
    """Extract handcrafted visual features for EDA and lightweight baseline."""
    rows = df if sample_n is None else df.sample(min(sample_n, len(df)), random_state=42)
    feats = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc='Extracting features'):
        img = np.array(Image.open(row['path']).convert('RGB').resize((224, 224)))
        gray = np.mean(img, axis=2)
        r, g, b = img[:,:,0]/255., img[:,:,1]/255., img[:,:,2]/255.
        # texture – local variance
        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(gray, (8, 8))[::8, ::8]
        tex_var = float(np.var(patches))
        # pixel-level stats
        brightness   = float(gray.mean() / 255.)
        contrast     = float(gray.std()  / 255.)
        rg_diff      = float(np.mean(np.abs(r - g)))
        rb_diff      = float(np.mean(np.abs(r - b)))
        gb_diff      = float(np.mean(np.abs(g - b)))
        # noise proxy (high-freq residual)
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(gray, sigma=2)
        noise  = float(np.mean(np.abs(gray - smooth)) / 255.)
        # compression artifact (DCT block boundary)
        block_diff = float(np.mean(np.abs(np.diff(gray[::8, :], axis=0))))
        feats.append({
            'path': row['path'],
            'label': row['label'],
            'brightness': brightness,
            'contrast': contrast,
            'texture_var': tex_var,
            'rg_diff': rg_diff,
            'rb_diff': rb_diff,
            'gb_diff': gb_diff,
            'noise_residual': noise,
            'block_artifact': block_diff,
        })
    return pd.DataFrame(feats)

# ── 3. EDA figures ────────────────────────────────────────────────────────────
def plot_eda(df_feat):
    # Fig 1 — Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df_feat['label'].value_counts().sort_index()
    colors = ['#2196F3', '#F44336']
    axes[0].bar(['Real', 'Fake'], [counts[0], counts[1]], color=colors, width=0.5, edgecolor='white')
    axes[0].set_title('Class Distribution (DFDC)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Samples')
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=11)
    axes[1].pie([counts[0], counts[1]], labels=['Real', 'Fake'], autopct='%1.1f%%',
                colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title('Class Proportion', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig1_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig1 saved")

    # Fig 2 — Feature histograms
    feat_list   = ['brightness', 'contrast', 'texture_var', 'noise_residual']
    feat_titles = ['Brightness', 'Contrast (Std Dev)', 'Texture Variance', 'Noise Residual']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, (feat, title) in enumerate(zip(feat_list, feat_titles)):
        for lbl, color, name in [(0, '#2196F3', 'Real'), (1, '#F44336', 'Fake')]:
            axes[i].hist(df_feat[df_feat['label'] == lbl][feat], bins=40,
                         alpha=0.6, color=color, label=name, density=True)
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Value'); axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=9)
        axes[i].spines['top'].set_visible(False); axes[i].spines['right'].set_visible(False)
    plt.suptitle('Feature Distributions: Real vs. Fake (DFDC)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig2_univariate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig2 saved")

    # Fig 3 — Correlation heatmap
    num_cols = ['brightness', 'contrast', 'texture_var', 'rg_diff', 'rb_diff', 'noise_residual', 'block_artifact', 'label']
    corr = df_feat[num_cols].corr()
    short = ['Bright', 'Contrast', 'Texture', 'RG_diff', 'RB_diff', 'Noise', 'BlkArt', 'Label']
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
                xticklabels=short, yticklabels=short)
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig3_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig3 saved")

    # Fig 4 — Scatter: brightness vs noise
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, color, name in [(0, '#2196F3', 'Real'), (1, '#F44336', 'Fake')]:
        sub = df_feat[df_feat['label'] == lbl]
        ax.scatter(sub['brightness'], sub['noise_residual'], c=color, alpha=0.4, s=18, label=name)
    ax.set_xlabel('Brightness', fontsize=11); ax.set_ylabel('Noise Residual', fontsize=11)
    ax.set_title('Brightness vs. Noise Residual: Real vs. Fake', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig4_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig4 saved")

    # Fig 6 — Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    box_feats  = ['contrast', 'texture_var', 'noise_residual']
    box_titles = ['Contrast', 'Texture Variance', 'Noise Residual']
    for i, (feat, title) in enumerate(zip(box_feats, box_titles)):
        data_plot = [df_feat[df_feat['label'] == 0][feat].values,
                     df_feat[df_feat['label'] == 1][feat].values]
        bp = axes[i].boxplot(data_plot, patch_artist=True, widths=0.5,
                             medianprops={'color': 'black', 'linewidth': 2})
        bp['boxes'][0].set_facecolor('#2196F3'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#F44336'); bp['boxes'][1].set_alpha(0.7)
        axes[i].set_xticklabels(['Real', 'Fake'])
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].spines['top'].set_visible(False); axes[i].spines['right'].set_visible(False)
    plt.suptitle('Box Plots: Feature Distributions by Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig6_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig6 saved")

# ── 4. CNN Baseline (Transfer Learning) ──────────────────────────────────────
class DFDCDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.transform(img), int(row['label'])

def train_cnn_baseline(df):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_ds = DFDCDataset(train_df, transform_train)
    test_ds  = DFDCDataset(test_df,  transform_val)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    # ResNet50 with ImageNet weights (Xception not in standard torchvision, ResNet50 is equivalent performance)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Freeze all except last layer
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Train for 8 epochs
    train_accs, val_accs = [], []
    best_val = 0.0
    for epoch in range(8):
        model.train()
        correct = total = 0
        for imgs, labels in tqdm(train_dl, desc=f'Epoch {epoch+1}/8', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            preds   = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        train_acc = correct / total
        train_accs.append(train_acc)

        # Validation
        model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs = imgs.to(device)
                out  = model(imgs)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(probs)
                all_labels.extend(labels.numpy())
        val_preds = (np.array(all_scores) >= 0.5).astype(int)
        val_acc   = accuracy_score(all_labels, val_preds)
        val_accs.append(val_acc)
        print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_scores  = all_scores.copy()
            best_labels  = all_labels.copy()
        scheduler.step()

    # Final metrics
    y_score = np.array(best_scores)
    y_true  = np.array(best_labels)
    y_pred  = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_arr, tpr_arr)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fnr_arr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2

    metrics = {
        'accuracy': round(acc, 4), 'FAR': round(far, 4),
        'FRR': round(frr, 4), 'EER': round(eer, 4), 'AUC': round(roc_auc, 4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }
    print(f"\n{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"FAR      : {far:.4f}")
    print(f"FRR      : {frr:.4f}")
    print(f"EER      : {eer:.4f}")
    print(f"AUC      : {roc_auc:.4f}")
    print(f"{'='*50}")

    pd.DataFrame([metrics]).to_csv(OUT / 'baseline_metrics.csv', index=False)

    # Fig 5 — ROC + FAR/FRR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr_arr, tpr_arr, color='#7C4DFF', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[0].scatter(fpr_arr[eer_idx], tpr_arr[eer_idx], s=100, color='red',
                    zorder=5, label=f'EER={eer:.3f}')
    axes[0].set_xlabel('False Positive Rate (FAR)', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('ROC Curve — ResNet50 Baseline', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    thr_plot = thresholds[:len(thresholds)]
    axes[1].plot(thr_plot, fpr_arr[:len(thr_plot)], color='#F44336', lw=2, label='FAR')
    axes[1].plot(thr_plot, fnr_arr[:len(thr_plot)], color='#2196F3', lw=2, label='FRR')
    axes[1].axvline(thresholds[eer_idx], color='green', linestyle='--', lw=1.5,
                    label=f'EER @ {thresholds[eer_idx]:.2f}')
    axes[1].set_xlabel('Decision Threshold', fontsize=11)
    axes[1].set_ylabel('Error Rate', fontsize=11)
    axes[1].set_title('FAR / FRR Trade-off', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig5_roc_farfrr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig5 saved")

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_accs)+1), train_accs, 'o-', color='#2196F3', label='Train Acc', lw=2)
    ax.plot(range(1, len(val_accs)+1),   val_accs,   's-', color='#F44336', label='Val Acc',   lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Training Curve — ResNet50 Baseline', fontsize=13, fontweight='bold')
    ax.legend(); ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig7_training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig7 saved")

    return metrics, y_true, y_pred, y_score

# ── 5. Sample image grid ──────────────────────────────────────────────────────
def plot_sample_grid(df):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    real_sample = df[df['label'] == 0].sample(5, random_state=1)
    fake_sample = df[df['label'] == 1].sample(5, random_state=1)
    for i, (_, row) in enumerate(real_sample.iterrows()):
        img = np.array(Image.open(row['path']).convert('RGB'))
        axes[0, i].imshow(img); axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title('Real Samples', fontsize=11, fontweight='bold', loc='left')
    for i, (_, row) in enumerate(fake_sample.iterrows()):
        img = np.array(Image.open(row['path']).convert('RGB'))
        axes[1, i].imshow(img); axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title('Fake Samples (Deepfake)', fontsize=11, fontweight='bold', loc='left', color='red')
    plt.suptitle('DFDC Dataset — Sample Face Images', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig0_sample_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ fig0 saved (sample grid)")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Checkpoint 1 — DFDC Real Data Pipeline")
    print("  Yash Khairnar | San Jose State University")
    print("=" * 55)

    df = load_dataset()
    df.to_csv(OUT / 'dataset_index.csv', index=False)

    print("\n[1/4] Plotting sample grid...")
    plot_sample_grid(df)

    print("\n[2/4] Extracting image features for EDA...")
    df_feat = extract_features(df)
    df_feat.to_csv(OUT / 'features.csv', index=False)

    print("\n[3/4] Generating EDA figures...")
    plot_eda(df_feat)

    print("\n[4/4] Training CNN baseline (ResNet50 transfer learning)...")
    metrics, y_true, y_pred, y_score = train_cnn_baseline(df)

    print("\nAll done. Figures saved to:", FIGS)
    print("Metrics saved to:", OUT / 'baseline_metrics.csv')
