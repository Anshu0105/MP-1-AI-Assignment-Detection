#!/usr/bin/env python3
"""
cross_val.py  –  Real 10-Fold Stratified Cross-Validation for Model C (CCRM)
Pools train + val + test embeddings (1,900 docs), trains a fresh CCRM from
scratch on each fold, and reports mean ± std for all key metrics.
Saves results to cross_val_results.json.
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModelC_CCRM

# ── Config ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "data")
TRAIN_PT   = os.path.join(DATA_DIR, "train_embeddings.pt")
VAL_PT     = os.path.join(DATA_DIR, "val_embeddings.pt")
TEST_PT    = os.path.join(DATA_DIR, "test_embeddings.pt")
OUT_JSON   = os.path.join(BASE, "cross_val_results.json")

N_FOLDS     = 10
EPOCHS      = 8
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY= 1e-4
HIDDEN_SIZE = 256
PATIENCE    = 3
SEED        = 42

# ── Device ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────
class SimpleSeqDataset(Dataset):
    def __init__(self, embeddings, lengths, labels):
        self.embeddings = embeddings
        self.lengths    = lengths
        self.labels     = labels
    def __len__(self):   return len(self.labels)
    def __getitem__(self, i):
        return self.embeddings[i], self.lengths[i], self.labels[i]

def collate(batch):
    embs, lens, labs = zip(*batch)
    max_len = max(e.shape[0] for e in embs)
    dim     = embs[0].shape[-1]
    padded  = torch.zeros(len(embs), max_len, dim)
    for i, e in enumerate(embs):
        padded[i, :e.shape[0]] = e
    return (padded,
            torch.tensor(lens, dtype=torch.long),
            torch.tensor(labs, dtype=torch.long))

def load_pt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d['embeddings'], d['lengths'], d['labels']

# ── Merge all splits ─────────────────────────────────────────────────────────
def load_full_pool():
    all_emb, all_len, all_lab = [], [], []
    for pt in [TRAIN_PT, VAL_PT, TEST_PT]:
        embs, lens, labs = load_pt(pt)
        all_emb.extend(embs)
        all_len.extend([int(l) for l in lens])
        all_lab.extend([int(l) for l in labs])
    all_len = torch.tensor(all_len, dtype=torch.long)
    all_lab = torch.tensor(all_lab, dtype=torch.long)
    print(f"Full pool: {len(all_emb)} docs  "
          f"(Human={int((all_lab==0).sum())}, AI={int((all_lab==1).sum())})")
    return all_emb, all_len, all_lab

# ── Train one epoch ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for emb, length, lab in loader:
        emb, length, lab = emb.to(device), length.to(device), lab.to(device)
        optimizer.zero_grad()
        logits = model(emb, length)
        loss   = criterion(logits, lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(lab)
    return total_loss / len(loader.dataset)

# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets, probs = 0.0, [], [], []
    with torch.no_grad():
        for emb, length, lab in loader:
            emb, length, lab = emb.to(device), length.to(device), lab.to(device)
            logits = model(emb, length)
            loss   = criterion(logits, lab)
            total_loss += loss.item() * len(lab)
            prob = torch.softmax(logits, -1)[:, 1]
            pred = torch.argmax(logits, -1)
            preds.extend(pred.cpu().tolist())
            targets.extend(lab.cpu().tolist())
            probs.extend(prob.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    acc  = accuracy_score(targets, preds)
    try:   auc = roc_auc_score(targets, probs)
    except: auc = 0.5
    cm = confusion_matrix(targets, preds).tolist()
    return dict(loss=avg_loss, accuracy=acc, precision=float(p),
                recall=float(r), f1=float(f1), roc_auc=float(auc),
                confusion_matrix=cm, preds=preds, targets=targets, probs=probs)

# ── Build fresh model ─────────────────────────────────────────────────────────
def build_model(device):
    m = ModelC_CCRM(
        embedding_dim=768, hidden_size=HIDDEN_SIZE,
        mlp_hidden=128, cognitive_dropout=0.2, classifier_dropout=0.3
    ).to(device)
    return m

# ════════════════════════════════════════════════════════════════════════════
# MAIN  10-FOLD CV
# ════════════════════════════════════════════════════════════════════════════
def run_cross_val():
    device    = get_device()
    print(f"\n{'='*65}")
    print(f"  10-FOLD STRATIFIED CROSS-VALIDATION — Model C (CCRM)")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    all_emb, all_len, all_lab = load_full_pool()
    full_dataset = SimpleSeqDataset(all_emb, all_len, all_lab)
    labels_np    = all_lab.numpy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np), 1):
        print(f"\n{'─'*55}")
        print(f"  FOLD {fold}/{N_FOLDS}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'─'*55}")

        train_sub = Subset(full_dataset, train_idx.tolist())
        val_sub   = Subset(full_dataset, val_idx.tolist())

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE,
                                  shuffle=True,  collate_fn=collate)
        val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=collate)

        # Class weights for this fold
        fold_labs  = labels_np[train_idx]
        n_human    = int((fold_labs == 0).sum())
        n_ai       = int((fold_labs == 1).sum())
        n_total    = len(fold_labs)
        w_human    = n_total / (2.0 * max(n_human, 1))
        w_ai       = n_total / (2.0 * max(n_ai,    1))
        weights    = torch.tensor([w_human, w_ai], dtype=torch.float32).to(device)
        criterion  = nn.CrossEntropyLoss(weight=weights)

        model      = build_model(device)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='max', factor=0.5, patience=2)

        best_f1   = 0.0
        best_state= None
        no_improve= 0
        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_m   = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_m['f1'])

            print(f"  Ep {epoch:02d}/{EPOCHS}  trainLoss={tr_loss:.4f}  "
                  f"valAcc={val_m['accuracy']*100:.2f}%  "
                  f"valF1={val_m['f1']:.4f}  valAUC={val_m['roc_auc']:.4f}")

            if val_m['f1'] > best_f1:
                best_f1   = val_m['f1']
                best_state= copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  [Early stop] No improvement for {PATIENCE} epochs.")
                    break

        # Re-evaluate with best weights
        model.load_state_dict(best_state)
        final_m = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        cm     = final_m['confusion_matrix']
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        fold_rec = {
            'fold':        fold,
            'val_size':    len(val_idx),
            'train_size':  len(train_idx),
            'accuracy':    round(final_m['accuracy'], 6),
            'precision':   round(final_m['precision'], 6),
            'recall':      round(final_m['recall'], 6),
            'f1':          round(final_m['f1'], 6),
            'roc_auc':     round(final_m['roc_auc'], 6),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'elapsed_sec': round(elapsed, 1)
        }
        fold_results.append(fold_rec)
        print(f"\n  ✅ Fold {fold} Done in {elapsed:.1f}s  |  "
              f"Acc={fold_rec['accuracy']:.4f}  "
              f"Prec={fold_rec['precision']:.4f}  "
              f"Rec={fold_rec['recall']:.4f}  "
              f"F1={fold_rec['f1']:.4f}  "
              f"AUC={fold_rec['roc_auc']:.4f}  "
              f"FP={fp}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  10-FOLD CV SUMMARY — Model C (CCRM)")
    print(f"{'='*65}")
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in fold_results]
        mean_ = float(np.mean(vals))
        std_  = float(np.std(vals))
        summary[k] = {'mean': round(mean_, 6), 'std': round(std_, 6),
                      'per_fold': [round(v, 6) for v in vals]}
        print(f"  {k:12s}:  {mean_:.4f} ± {std_:.4f}  "
              f"[min={min(vals):.4f}, max={max(vals):.4f}]")

    total_fp = sum(r['FP'] for r in fold_results)
    total_fn = sum(r['FN'] for r in fold_results)
    print(f"\n  Total FP across all folds : {total_fp}")
    print(f"  Total FN across all folds : {total_fn}")

    output = {
        'model':        'Model C (CCRM)',
        'n_folds':      N_FOLDS,
        'total_docs':   len(all_emb),
        'seed':         SEED,
        'summary':      summary,
        'fold_results': fold_results,
        'total_FP':     total_fp,
        'total_FN':     total_fn,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {OUT_JSON}\n")
    return output

if __name__ == "__main__":
    run_cross_val()
