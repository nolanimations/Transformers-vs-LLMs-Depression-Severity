"""
BiLSTM floor baseline with pre-trained word embeddings (GloVe 300d recommended).
Config-driven via configs/lstm.yaml.

  1. Build a word vocabulary from the training set texts
  2. Load GloVe embeddings (download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip)
  3. Define a Bidirectional LSTM model in PyTorch
  4. Write a training loop with early stopping (patience is in the config)
  5. Use class weights from src.data.class_weights() — severe is only 6% of the data
  6. Call src.eval.evaluate() on the test set when done

Usage:
    python src/lstm_baseline.py --config configs/lstm.yaml

See notebooks/03_lstm_run.ipynb for step-by-step guidance.
"""
#------------------------------------------------------------------------------------
import sys
from pathlib import Path

# When running the script as `python src/lstm_baseline.py` the import
# machinery sets sys.path[0] to the `src/` directory which prevents
# `import src` from resolving. Ensure the repo root is on sys.path so
# `from src import ...` works whether run as a script or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#------------------------------------------------------------------------------------

import argparse
import yaml
from src.data import load_splits, class_weights
from src.eval import evaluate
from src.utils import set_seed, get_logger
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

log = get_logger(__name__)


class TextDataset(Dataset):
    def __init__(self, df, word2idx, max_len=128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = str(self.texts[idx]).lower().split()
        ids = [self.word2idx.get(t, self.word2idx.get("<unk>")) for t in toks]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        return ids, self.labels[idx]


def collate_batch(batch, pad_idx=0):
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
    max_l = int(lengths.max()) if len(lengths) > 0 else 0
    padded = torch.full((len(ids_list), max_l), pad_idx, dtype=torch.long)
    for i, seq in enumerate(ids_list):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


class BiLSTMModel(nn.Module):
    def __init__(self, emb_tensor, cfg, pad_idx=0, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor, freeze=cfg.get("freeze_embeddings", False), padding_idx=pad_idx
        )
        self.hidden = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.bidirectional = cfg.get("bidirectional", True)
        self.lstm = nn.LSTM(
            input_size=emb_tensor.shape[1],
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=cfg.get("dropout", 0.0) if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        out_dim = self.hidden * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        lengths_sorted, perm_idx = lengths.sort(descending=True)
        emb_sorted = emb[perm_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        if self.bidirectional:
            forward = h_n[-2]
            backward = h_n[-1]
            h = torch.cat([forward, backward], dim=1)
        else:
            h = h_n[-1]
        _, unperm_idx = perm_idx.sort()
        h = h[unperm_idx]
        h = self.dropout(h)
        logits = self.fc(h)
        return logits


def train_and_eval(
    model,
    opt,
    criterion,
    train_loader,
    val_loader,
    test_loader,
    cfg,
    device,
    out_dir: Path,
    test_df=None,
    resume: bool = False,
    use_amp: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pt"

    start_epoch = 1
    best_val = -1.0
    if resume and best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state", {}))
        if "opt_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt_state"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_f1", best_val)

    patience = cfg.get("early_stopping_patience", 3)
    no_improve = 0
    num_epochs = cfg.get("num_epochs", 10)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch in train_loader:
            input_ids, lengths, labels = batch
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, lengths)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            running_loss += loss.item() * input_ids.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, lengths, labels = batch
                input_ids = input_ids.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, lengths)
                else:
                    logits = model(input_ids, lengths)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{num_epochs} — train_loss={train_loss:.4f} val_macro_f1={val_f1:.4f} time={elapsed:.1f}s")

        if val_f1 > best_val:
            best_val = val_f1
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "epoch": epoch, "val_f1": val_f1}, best_ckpt)
            print("Saved best checkpoint", best_ckpt)
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{patience})")
        if no_improve >= patience:
            print("Early stopping")
            break

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"]) 
    model.to(device)

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids, lengths, labels = batch
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, lengths)
            else:
                logits = model(input_ids, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    try:
        if test_df is not None:
            evaluate(preds=np.array(all_preds), labels=np.array(all_labels), df=test_df, run_name=cfg.get("run_name","lstm"), output_dir=str(out_dir), split="test")
            print("Finished evaluation. Results saved to", out_dir)
    except Exception as exc:
        print("Evaluation failed:", exc)

    return best_ckpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    # Prepare data, vocab and embeddings (expects `data/vocab.json` and `data/glove_embeddings.pt` exist)
    train_df, val_df, test_df = load_splits()

    vocab_path = Path("data") / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found at {vocab_path} — run the notebook vocab cell first")
    word2idx = json.loads(vocab_path.read_text(encoding="utf8"))["word2idx"]
    pad_idx = word2idx.get("<pad>", 0)

    emb_file = Path("data") / "glove_embeddings.pt"
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_file} — run the notebook embedding cell first")
    emb_tensor = torch.load(emb_file)

    # datasets + loaders
    train_ds = TextDataset(train_df, word2idx, max_len=cfg.get("max_length", 128))
    val_ds = TextDataset(val_df, word2idx, max_len=cfg.get("max_length", 128))
    test_ds = TextDataset(test_df, word2idx, max_len=cfg.get("max_length", 128))

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, pad_idx=pad_idx))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_idx=pad_idx))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_idx=pad_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel(emb_tensor, cfg, pad_idx=pad_idx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-3))
    weights = class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    out_dir = Path(cfg.get("output_dir", "results/runs/lstm"))

    # run training & evaluation
    train_and_eval(
        model=model,
        opt=opt,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        test_df=test_df,
        resume=False,
        use_amp=False,
    )
