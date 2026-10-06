#!/usr/bin/env python3
"""
benchmark_paper.py  –  Real zero-shot cross-dataset evaluation on the
paper's (Prajapati et al., 2026) underlying Kaggle benchmark CSVs.

Runs the trained model_c_ccrm.pth checkpoint (no retraining) against
50 Persuade human essays + 50 AI essays from Claude, MOTH, PaLM,
Falcon-180B, LLaMA-70B.

Saves: benchmark_paper_results.json
"""

import os, sys, json, re
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModelC_CCRM, masked_mean_pooling

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
CKPT    = os.path.join(BASE, "model_c_ccrm.pth")
BENCH   = os.path.join(BASE, "..", "Benchmark_Datasets")
OUT_JSON= os.path.join(BASE, "benchmark_paper_results.json")

DEBERTA_MODEL = "microsoft/deberta-v3-base"
MAX_PARA_LEN  = 512      # tokens per paragraph
SAMPLES_EACH  = 10       # AI samples per source
HUMAN_SAMPLES = 50       # human samples from Persuade

# ── Device ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")

# ── Segment text into paragraphs ─────────────────────────────────────────────
def segment_paragraphs(text: str):
    paras = [p.strip() for p in re.split(r'\n\n+|\n(?=[A-Z])', text) if p.strip()]
    return paras if paras else [text.strip()]

# ── Embed one document ────────────────────────────────────────────────────────
@torch.no_grad()
def embed_document(text, tokenizer, encoder, device):
    paras = segment_paragraphs(text)
    para_embs = []
    for para in paras:
        enc = tokenizer(
            para, return_tensors='pt', truncation=True,
            max_length=MAX_PARA_LEN, padding='max_length'
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = masked_mean_pooling(out.last_hidden_state, attention_mask)  # [1, 768]
        para_embs.append(emb.squeeze(0).cpu())
    stacked = torch.stack(para_embs)  # [n_paras, 768]
    return stacked, len(paras)

# ── Run inference on a list of texts ─────────────────────────────────────────
@torch.no_grad()
def run_inference(texts, labels, model, tokenizer, encoder, device, source_name):
    model.eval()
    preds, probs, targets = [], [], []
    print(f"  [{source_name}] Embedding {len(texts)} docs...", end=' ', flush=True)
    for i, text in enumerate(texts):
        emb, n_para = embed_document(text, tokenizer, encoder, device)
        emb_dev  = emb.unsqueeze(0).to(device)   # [1, n_para, 768]
        length   = torch.tensor([n_para], dtype=torch.long, device=device)
        logits   = model(emb_dev, length)
        prob     = torch.softmax(logits, -1)[0, 1].item()
        pred     = int(torch.argmax(logits, -1).item())
        preds.append(pred)
        probs.append(prob)
        targets.append(labels[i])
        if (i+1) % 10 == 0:
            print(f"{i+1}", end=' ', flush=True)
    print("✓")

    acc  = accuracy_score(targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    try:   auc = roc_auc_score(targets, probs)
    except: auc = float('nan')
    cm   = confusion_matrix(targets, preds, labels=[0,1]).tolist()
    tp   = cm[1][1]; tn = cm[0][0]; fp = cm[0][1]; fn = cm[1][0]

    return {
        'source':    source_name,
        'n_docs':    len(texts),
        'accuracy':  round(acc, 4),
        'precision': round(float(p), 4),
        'recall':    round(float(r), 4),
        'f1':        round(float(f1), 4),
        'roc_auc':   round(float(auc), 4) if not np.isnan(auc) else 'n/a',
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'avg_prob_ai': round(float(np.mean([p_ for p_, l in zip(probs, targets) if l == 1])) if any(l==1 for l in targets) else 0.0, 4),
        'avg_prob_human': round(float(np.mean([p_ for p_, l in zip(probs, targets) if l == 0])) if any(l==0 for l in targets) else 0.0, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark():
    device = get_device()
    print(f"\n{'='*65}")
    print(f"  ZERO-SHOT BENCHMARK vs. Prajapati et al. (2026) Kaggle Datasets")
    print(f"  Model C (CCRM) — No Retraining")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading Model C checkpoint...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt.get('config', {})
    model = ModelC_CCRM(
        embedding_dim=cfg.get('embedding_dim', 768),
        hidden_size=cfg.get('hidden_size', 256),
        mlp_hidden=cfg.get('mlp_hidden', 128),
        cognitive_dropout=cfg.get('cognitive_dropout', 0.2),
        classifier_dropout=cfg.get('classifier_dropout', 0.3)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Checkpoint loaded (epoch {ckpt.get('epoch','?')})")

    # ── Load DeBERTa ──────────────────────────────────────────────────────────
    print(f"\nLoading DeBERTa tokenizer & encoder ({DEBERTA_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
    encoder   = AutoModel.from_pretrained(DEBERTA_MODEL).to(device)
    encoder.eval()
    print("  ✓ Encoder loaded (FROZEN)")

    # ── Load benchmark data ───────────────────────────────────────────────────
    all_results = []

    # 1. HUMAN — Human_story column from Human_and_diff_AIs.csv (project's real human essays)
    print("\n[1/6] Loading Human essays (Human_and_diff_AIs.csv → Human_story)...")
    proj_csv = os.path.join(BASE, "..", "Project Work", "Human_and_diff_AIs.csv")
    human_df = pd.read_csv(proj_csv, usecols=['Human_story'], low_memory=False)
    human_texts = [t for t in human_df['Human_story'].dropna().astype(str).tolist() if len(t.strip()) > 100]
    human_texts = human_texts[:HUMAN_SAMPLES]
    print(f"  ✓ {len(human_texts)} human essays loaded")
    r_human = run_inference(human_texts, [0]*len(human_texts), model, tokenizer, encoder, device, "Persuade Corpus (Human)")
    all_results.append(r_human)

    # 2. CLAUDE — essay_text column (persuade15_claude_instant1.csv: all 1000 rows = Claude AI)
    print("\n[2/6] Loading Claude Instant v1 essays (essay_text column, all AI)...")
    claude_csv   = pd.read_csv(os.path.join(BENCH, "claude", "persuade15_claude_instant1.csv"))
    claude_texts = claude_csv['essay_text'].dropna().astype(str).head(SAMPLES_EACH).tolist()
    print(f"  ✓ {len(claude_texts)} Claude essays loaded")
    r_claude = run_inference(claude_texts, [1]*len(claude_texts), model, tokenizer, encoder, device, "Claude Instant v1 (AI)")
    all_results.append(r_claude)

    # 3. MOTH / ChatGPT — text column (all AI-generated)
    print("\n[3/6] Loading MOTH / ChatGPT essays (text column, all AI)...")
    moth_csv   = pd.read_csv(os.path.join(BENCH, "moth", "daigt_external_dataset.csv"))
    moth_texts = moth_csv['text'].dropna().astype(str).head(SAMPLES_EACH).tolist()
    print(f"  ✓ {len(moth_texts)} MOTH essays loaded")
    r_moth = run_inference(moth_texts, [1]*len(moth_texts), model, tokenizer, encoder, device, "MOTH / ChatGPT (AI)")
    all_results.append(r_moth)

    # 4. PaLM-Bison — text column (generated==1, all AI)
    print("\n[4/6] Loading PaLM-Bison essays (text column, all AI)...")
    palm_csv   = pd.read_csv(os.path.join(BENCH, "palm", "LLM_generated_essay_PaLM.csv"))
    palm_texts = palm_csv['text'].dropna().astype(str).head(SAMPLES_EACH).tolist()
    print(f"  ✓ {len(palm_texts)} PaLM essays loaded")
    r_palm = run_inference(palm_texts, [1]*len(palm_texts), model, tokenizer, encoder, device, "PaLM-Bison (AI)")
    all_results.append(r_palm)

    # 5. Falcon 180B
    print("\n[5/6] Loading Falcon 180B essays (AI)...")
    falcon_csv  = pd.read_csv(os.path.join(BENCH, "llama_falcon", "falcon_180b_v1.csv"))
    falcon_text = 'text' if 'text' in falcon_csv.columns else falcon_csv.columns[0]
    falcon_texts = falcon_csv[falcon_text].dropna().astype(str).head(SAMPLES_EACH).tolist()
    print(f"  ✓ {len(falcon_texts)} Falcon essays loaded")
    r_falcon = run_inference(falcon_texts, [1]*len(falcon_texts), model, tokenizer, encoder, device, "Falcon 180B (AI)")
    all_results.append(r_falcon)

    # 6. LLaMA 70B
    print("\n[6/6] Loading LLaMA 70B essays (AI)...")
    llama_csv  = pd.read_csv(os.path.join(BENCH, "llama_falcon", "llama_70b_v1.csv"))
    llama_text = 'text' if 'text' in llama_csv.columns else llama_csv.columns[0]
    llama_texts = llama_csv[llama_text].dropna().astype(str).head(SAMPLES_EACH).tolist()
    print(f"  ✓ {len(llama_texts)} LLaMA essays loaded")
    r_llama = run_inference(llama_texts, [1]*len(llama_texts), model, tokenizer, encoder, device, "LLaMA 70B (AI)")
    all_results.append(r_llama)

    # ── Overall aggregate ─────────────────────────────────────────────────────
    total_correct = sum(r['TP'] + r['TN'] for r in all_results)
    total_docs    = sum(r['n_docs'] for r in all_results)
    total_tp  = sum(r['TP'] for r in all_results)
    total_tn  = sum(r['TN'] for r in all_results)
    total_fp  = sum(r['FP'] for r in all_results)
    total_fn  = sum(r['FN'] for r in all_results)
    overall_acc = total_correct / total_docs if total_docs > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  OVERALL CROSS-DATASET RESULTS")
    print(f"{'='*65}")
    print(f"  Total docs : {total_docs}  (Human={HUMAN_SAMPLES}, AI={total_docs - HUMAN_SAMPLES})")
    print(f"  Correct    : {total_correct} / {total_docs}")
    print(f"  Accuracy   : {overall_acc*100:.2f}%")
    print(f"  TP={total_tp}  TN={total_tn}  FP={total_fp}  FN={total_fn}")
    print()
    print(f"  {'Source':<30}  {'n':>4}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*75}")
    for r in all_results:
        print(f"  {r['source']:<30}  {r['n_docs']:>4}  "
              f"{r['accuracy']*100:>6.1f}%  "
              f"{r['precision']:>6.3f}   "
              f"{r['recall']:>6.3f}   "
              f"{r['FP']:>4}  {r['FN']:>4}")

    overall = {
        'total_docs':   total_docs,
        'total_correct':total_correct,
        'overall_accuracy': round(overall_acc, 4),
        'TP': total_tp, 'TN': total_tn, 'FP': total_fp, 'FN': total_fn,
        'precision': round(total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0, 4),
        'recall':    round(total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0, 4),
    }

    output = {
        'model':          'Model C (CCRM)',
        'source_results': all_results,
        'overall':        overall,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {OUT_JSON}\n")
    return output

if __name__ == "__main__":
    run_benchmark()
