import os
import json
import glob
import random
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import kagglehub

# ==========================================
# Configuration & Globals
# ==========================================
CACHE_DIR = "./cache"
SAVE_DIR = "./avfake_results"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")

N_FRAMES  = 16
SR        = 16000
N_MELS    = 128
DURATION  = 3.0
IMG_SIZE  = 224
MAX_CLIPS = 5000
DIM       = 512
BATCH     = 8
WARMUP_EPS = 5
TOTAL_EPS  = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225])
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_mels=N_MELS, n_fft=1024, hop_length=160)
amp2db = torchaudio.transforms.AmplitudeToDB()

# ==========================================
# Models
# ==========================================
class FaceEncoder(nn.Module):
    def __init__(self, out_dim=DIM, freeze=True):
        super().__init__()
        self.vit  = timm.create_model('vit_small_patch16_224',
                                       pretrained=True, num_classes=0)
        self.proj = nn.Linear(384, out_dim)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.vit(x.view(B*T, C, H, W))
        return self.proj(f).view(B, T, -1)
        
    def unfreeze(self):
        for p in self.vit.parameters():
            p.requires_grad = True

class VoiceEncoder(nn.Module):
    def __init__(self, out_dim=DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc = nn.Linear(128*4*4, out_dim)
        
    def forward(self, x):
        return self.fc(self.cnn(x).view(x.size(0), -1))

class CrossModalAttention(nn.Module):
    def __init__(self, dim=DIM, heads=8, dropout=0.1):
        super().__init__()
        self.attn_v2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_a2v = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_v   = nn.LayerNorm(dim)
        self.norm_a   = nn.LayerNorm(dim)
        self.ffn_v    = nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*2,dim))
        self.ffn_a    = nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*2,dim))
        self.norm_v2  = nn.LayerNorm(dim)
        self.norm_a2  = nn.LayerNorm(dim)
        
    def forward(self, fv, fa):
        fa_exp   = fa.unsqueeze(1).expand_as(fv)
        v_att, _ = self.attn_v2a(fv, fa_exp, fa_exp)
        v_ctx    = self.norm_v2(self.norm_v(fv+v_att) + self.ffn_v(self.norm_v(fv+v_att)))
        a_att, _ = self.attn_a2v(fa_exp, fv, fv)
        a_ctx    = self.norm_a2(self.norm_a(fa_exp+a_att) + self.ffn_a(self.norm_a(fa_exp+a_att)))
        return v_ctx, a_ctx

class AVConsist(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.face_enc   = FaceEncoder(freeze=freeze)
        self.voice_enc  = VoiceEncoder()
        self.cross_attn = CrossModalAttention()
        self.classifier = nn.Sequential(
            nn.Linear(DIM*2, 256), nn.ReLU(),
            nn.Dropout(0.3),       nn.Linear(256, 1))
            
    def forward(self, frames, mel):
        fv = self.face_enc(frames)
        fa = self.voice_enc(mel)
        v_ctx, a_ctx = self.cross_attn(fv, fa)
        fused = torch.cat([v_ctx.mean(1), a_ctx.mean(1)], dim=-1)
        return self.classifier(fused).squeeze(-1)
        
    def unfreeze_encoders(self):
        self.face_enc.unfreeze()
        print("Encoders unfrozen.")

# ==========================================
# Dataset & Preprocessing
# ==========================================
class AVDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        # Setting weights_only=False to allow loading the custom dictionary structure
        d = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        return d['faces'], d['mel'], torch.tensor(d['label'], dtype=torch.float32)

def extract_frames(video_path):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None
        
    indices = np.linspace(0, max(total-1,0), N_FRAMES, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rs  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        t   = torch.from_numpy(rs).permute(2,0,1).float() / 255.0
        frames.append(normalize(t))
    cap.release()
    
    if not frames: 
        return None
    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    return torch.stack(frames[:N_FRAMES])

def extract_mel(video_path):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
        
    r = subprocess.run(
        ['ffmpeg','-i',str(video_path),'-ar',str(SR),'-ac','1','-y',tmp],
        capture_output=True)
        
    if r.returncode != 0:
        os.unlink(tmp)
        return None
        
    try:
        waveform, _ = torchaudio.load(tmp)
    except:
        if os.path.exists(tmp): os.unlink(tmp)
        return None
        
    if os.path.exists(tmp): 
        os.unlink(tmp)
        
    target = int(SR * DURATION)
    if waveform.shape[1] < target:
        waveform = torch.nn.functional.pad(
            waveform, (0, target - waveform.shape[1]))
    else:
        waveform = waveform[:, :target]
    return amp2db(mel_tf(waveform))

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, all_scores, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    
    with ctx:
        for frames, mel, labels in loader:
            frames, mel, labels = frames.to(DEVICE), mel.to(DEVICE), labels.to(DEVICE)
            
            if train: 
                optimizer.zero_grad()
                
            logits = model(frames, mel)
            loss   = criterion(logits, labels)
            
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            total_loss += loss.item()
            all_scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    s, l = np.array(all_scores), np.array(all_labels)
    acc  = ((s>=0.5).astype(int)==l).mean()*100
    fpr, tpr, _ = roc_curve(l,s)
    
    # Handle cases where batch might only have one class
    if len(np.unique(l)) > 1:
        rauc = sk_auc(fpr,tpr)
        fnr  = 1-tpr
        ei   = np.nanargmin(np.abs(fnr-fpr))
        eer  = (fpr[ei]+fnr[ei])/2*100
    else:
        rauc, eer = 0.0, 0.0
        
    return total_loss/len(loader), acc, rauc, eer

# ==========================================
# Main Execution
# ==========================================
def main(args):
    print("="*50)
    print("Audio-Visual Deepfake Detection Training")
    print("="*50)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Download and Parse Metadata
    print("\n[1/5] Downloading / Finding Dataset...")
    dataset_path = kagglehub.dataset_download("elin75/localized-audio-visual-deepfake-dataset-lav-df")
    print(f"Dataset path: {dataset_path}")
    base_dir = os.path.join(dataset_path, "LAV-DF")

    with open(os.path.join(base_dir, "metadata.json")) as f:
        meta_raw = json.load(f)

    clip_labels = {}
    for entry in meta_raw:
        if not isinstance(entry, dict):
            continue
        fpath = entry.get('file', '')
        if entry.get('fake', False) or entry.get('label','') == 'fake':
            lbl = 1
        elif entry.get('modify_video') or entry.get('modify_audio'):
            lbl = 1
        else:
            lbl = 0
        clip_labels[fpath] = lbl

    real_count = sum(1 for l in clip_labels.values() if l==0)
    fake_count = sum(1 for l in clip_labels.values() if l==1)
    print(f"Parsed metadata: {len(clip_labels)} clips (Real: {real_count} | Fake: {fake_count})")

    # 2. Extract Features
    print("\n[2/5] Extracting Features & Caching...")
    real_clips = [(Path(base_dir) / k, v) for k, v in clip_labels.items() if v == 0 and (Path(base_dir) / k).exists()]
    fake_clips = [(Path(base_dir) / k, v) for k, v in clip_labels.items() if v == 1 and (Path(base_dir) / k).exists()]

    random.seed(42)
    random.shuffle(real_clips)
    random.shuffle(fake_clips)

    use_clips = real_clips[:MAX_CLIPS] + fake_clips[:MAX_CLIPS]
    random.shuffle(use_clips)
    
    ok, failed = 0, 0
    for i, (vpath, label) in enumerate(tqdm(use_clips, desc="Caching Clips")):
        out = Path(args.cache_dir) / f"{vpath.stem}.pt"
        if out.exists():
            ok += 1; continue

        faces = extract_frames(vpath)
        mel   = extract_mel(vpath)

        if faces is None or mel is None:
            failed += 1; continue

        torch.save({'faces': faces, 'mel': mel, 'label': label}, out)
        ok += 1

    cached_files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    print(f"Caching complete. Cached: {len(cached_files)} | Failed: {failed}")
    if len(cached_files) == 0:
        print("No files cached. Exiting.")
        return

    # 3. Prepare Dataloaders
    print("\n[3/5] Preparing Dataloaders...")
    labels_all = [torch.load(f, map_location='cpu', weights_only=False)['label'] for f in cached_files]
    n_real = sum(1 for l in labels_all if l == 0)
    n_fake = sum(1 for l in labels_all if l == 1)
    
    train_f, temp_f = train_test_split(cached_files, test_size=0.30, random_state=42, stratify=labels_all)
    lab_temp        = [torch.load(f, map_location='cpu', weights_only=False)['label'] for f in temp_f]
    val_f, test_f   = train_test_split(temp_f, test_size=0.50, random_state=42, stratify=lab_temp)

    train_loader = DataLoader(AVDataset(train_f), batch_size=BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(AVDataset(val_f),   batch_size=BATCH, shuffle=False, num_workers=2)
    test_loader  = DataLoader(AVDataset(test_f),  batch_size=BATCH, shuffle=False, num_workers=2)
    print(f"Splits - Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

    # 4. Model Training
    print("\n[4/5] Training Model...")
    model = AVConsist(freeze=True).to(DEVICE)
    pos_weight = torch.tensor([n_real / max(n_fake,1)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPS)

    history, best_eer = [], 999.0
    for epoch in range(1, TOTAL_EPS+1):
        if epoch == WARMUP_EPS+1:
            model.unfreeze_encoders()
            # Re-initialize optimizer to include the newly unfrozen parameters properly
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPS-WARMUP_EPS)

        tr_loss, tr_acc, tr_auc, tr_eer = run_epoch(model, train_loader, optimizer, criterion, train=True)
        vl_loss, vl_acc, vl_auc, vl_eer = run_epoch(model, val_loader, optimizer, criterion, train=False)
        scheduler.step()

        print(f"Ep {epoch:02d}/{TOTAL_EPS} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.1f}% eer={tr_eer:.2f}% | "
              f"Val loss={vl_loss:.4f} acc={vl_acc:.1f}% eer={vl_eer:.2f}%")

        history.append(dict(epoch=epoch, tr_loss=round(tr_loss,4), tr_acc=round(tr_acc,2),
                            tr_eer=round(tr_eer,2), vl_loss=round(vl_loss,4),
                            vl_acc=round(vl_acc,2), vl_eer=round(vl_eer,2)))
        
        if vl_eer < best_eer:
            best_eer = vl_eer
            torch.save(model.state_dict(), args.model_path)

    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 5. Evaluation
    print("\n[5/5] Evaluating Best Model...")
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for frames, mel, labels in test_loader:
            logits = model(frames.to(DEVICE), mel.to(DEVICE))
            all_scores.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    y_true = np.array(all_labels)
    y_pred = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/len(y_true)*100
    far = fp/(fp+tn)*100 if (fp+tn)>0 else 0
    frr = fn/(fn+tp)*100 if (fn+tp)>0 else 0

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = sk_auc(fpr, tpr)
    fnr     = 1-tpr
    eer_idx = np.nanargmin(np.abs(fnr-fpr))
    eer     = (fpr[eer_idx]+fnr[eer_idx])/2*100

    print("\n" + "="*50)
    print("AVConsist — Test Results on LAV-DF")
    print("="*50)
    print(f"Accuracy : {acc:.2f}%")
    print(f"FAR      : {far:.2f}%")
    print(f"FRR      : {frr:.2f}%")
    print(f"EER      : {eer:.2f}%")
    print(f"AUC      : {roc_auc:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Real','Fake']))

    metrics = dict(accuracy=round(acc,2), FAR=round(far,2), FRR=round(frr,2),
                   EER=round(eer,2), AUC=round(roc_auc,4),
                   TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))
    
    with open(os.path.join(args.save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot ROC & FAR/FRR
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot(fpr, tpr, color='#26A69A', lw=2, label=f'AUC={roc_auc:.4f}')
    axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    axes[0].scatter(fpr[eer_idx], tpr[eer_idx], s=80, color='red', zorder=5, label=f'EER={eer:.2f}%')
    axes[0].set_xlabel('FAR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve — AVConsist', fontweight='bold')
    axes[0].legend(); axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    thr_v = thresholds[:len(thresholds)]
    axes[1].plot(thr_v, fpr[:len(thr_v)]*100, color='#EF5350', lw=2, label='FAR')
    axes[1].plot(thr_v, fnr[:len(thr_v)]*100, color='#42A5F5', lw=2, label='FRR')
    axes[1].axvline(thr_v[eer_idx], color='green', linestyle='--', lw=1.5, label=f'EER@{thr_v[eer_idx]:.2f}')
    axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Error Rate (%)')
    axes[1].set_title('FAR/FRR Trade-off', fontweight='bold')
    axes[1].legend(); axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'roc_farfrr.png'), dpi=150, bbox_inches='tight')
    print(f"\nAll results, metrics, and plots saved to '{args.save_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train AVConsist on LAV-DF")
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR, help='Directory to cache preprocessed features')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, help='Directory to save results and plots')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH, help='Path to save best model')
    args = parser.parse_args()
    
    main(args)
