"""LoRA fine-tune of the BGE cross-encoder reranker — CPU-friendly.

Trains a small LoRA adapter on top of BAAI/bge-reranker-base using the
(query, passage, label) pairs. Tiny rank, few epochs, small batch → runs on
CPU in minutes. Cross-encoder = sequence-classification (1 logit); trained as
regression toward 1.0/0.0 labels with BCE-with-logits.

Output: models/reranker-lora/

Usage:
    python -m scripts.train_reranker_lora
    python -m scripts.train_reranker_lora --epochs 3 --rank 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "BAAI/bge-reranker-base"
TRAIN = Path("eval/rerank/train_pairs.jsonl")
OUT = Path("models/reranker-lora")


class PairDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["query"], r["passage"], float(r["label"])


def collate(batch, tokenizer, max_len=256):
    queries = [b[0] for b in batch]
    passages = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float)
    enc = tokenizer(
        queries, passages,
        padding=True, truncation=True, max_length=max_len, return_tensors="pt",
    )
    return enc, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not TRAIN.exists():
        print(f"ERROR: {TRAIN} not found. Run scripts.make_rerank_dataset first.")
        return 1

    torch.manual_seed(42)
    device = "cpu"
    print(f"Loading base model {BASE_MODEL} (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    ds = PairDataset(TRAIN)
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer),
    )
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"Training: {len(ds)} pairs, {args.epochs} epochs, rank {args.rank}...")
    for epoch in range(args.epochs):
        total = 0.0
        for enc, labels in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            out = model(**enc)
            logits = out.logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch+1}/{args.epochs}  loss={total/len(dl):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"Saved LoRA adapter -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())