#!/usr/bin/env python3
"""
PHASE 3 (v2) - Fine-tune vinai/phobert-base-v2 for 3-class sentiment and label the full corpus
=============================================================================================
Inputs (current folder by default - ideal for Colab)
    gold_labeled.csv    adjudicated ground truth from 02c: needs  review_id, label  (Positive|Negative|Neutral)
    reviews_clean.csv   full corpus (12,926 rows): needs review_id and text_seg (word-segmented text)

Outputs (folder phobert_outputs/)
    reviews_phobert_predicted.csv   the FULL reviews_clean.csv + pred_label, p_neg, p_neu, p_pos, confidence,
                                    split (train/val/test/unlabelled), sent_final
    metrics_summary.json            accuracy, macro-F1, Cohen's kappa (per seed + ensemble), bootstrap CI, baseline
    confusion_matrix_test.csv/.png  counts and row-normalised
    classification_report_test.csv  per-class precision / recall / F1
    errors_test.csv                 misclassified test reviews (for the qualitative error analysis)
    splits.csv                      frozen review_id -> split map (never re-split)

Colab (Runtime > Change runtime type > T4 GPU), then:
    !pip install -q "transformers>=4.40" accelerate scikit-learn matplotlib statsmodels
    !python 03_phobert_training_v2.py --gold gold_labeled.csv --clean reviews_clean.csv

Design decisions
  * Input = text_seg (PhoBERT was pre-trained on word-segmented Vietnamese). max_len 128; the script prints the
    95/99th percentile token length so the choice can be justified.
  * Split 70/15/15 stratified by label x App; the split is frozen (--split-seed). The test split is scored once per
    seed after model selection on the validation split.
  * Class imbalance: class-weighted cross-entropy, weight_c = (N / (K * n_c)) ** --weight-power (computed on the
    TRAIN split only; power 1 = fully balanced, 0.5 = softer). Model selection uses macro-F1, not accuracy.
  * T4 settings: fp16, batch 16, lr 2e-5, up to 8 epochs with early stopping (patience 2), 10% warm-up,
    weight decay 0.01, 3 seeds averaged (probability ensemble).
  * Rows used in training/validation keep their HUMAN label in `sent_final`; test and unlabelled rows keep the model
    label (so no prediction on training data is presented as a model result).
"""
import argparse
import inspect
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

LABELS = ["Negative", "Neutral", "Positive"]
L2I = {l: i for i, l in enumerate(LABELS)}


# --------------------------------------------------------------------------- #
class TextDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels=None):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights (signature tolerant to transformers 4.x / 5.x)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1_macro": f1_score(p.label_ids, preds, average="macro"),
            "kappa": cohen_kappa_score(p.label_ids, preds)}


def make_args(outdir, a, seed):
    kw = dict(output_dir=str(outdir), learning_rate=a.lr, per_device_train_batch_size=a.bs,
              per_device_eval_batch_size=64, num_train_epochs=a.epochs, weight_decay=0.01,
              seed=seed, fp16=torch.cuda.is_available(), save_strategy="epoch", logging_steps=50,
              load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
              save_total_limit=1, report_to="none", disable_tqdm=False)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = 0.1
    else:
        kw["warmup_steps"] = 0.1
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**kw)


def make_trainer(model, args, train_ds, val_ds, tok, weights):
    kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
              data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
              callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], class_weights=weights)
    params = inspect.signature(Trainer.__init__).parameters
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tok
    return WeightedTrainer(**kw)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def bootstrap_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    stats = [f1_score(y[i], p[i], average="macro") for i in (rng.randint(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "cohen_kappa": float(cohen_kappa_score(y, pred))}


def mcnemar_exact(y, p1, p2):
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        return None
    c1, c2 = p1 == y, p2 == y
    tab = [[int((c1 & c2).sum()), int((c1 & ~c2).sum())], [int((~c1 & c2).sum()), int((~c1 & ~c2).sum())]]
    return {"table": tab, "p_value": float(mcnemar(tab, exact=True).pvalue)}


def plot_cm(y, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, p, labels=range(3))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a_, m, t, f in ((ax[0], cm, "Counts", "d"), (ax[1], cmn, "Row-normalised (recall)", ".2f")):
        a_.imshow(m, cmap="Blues")
        a_.set_xticks(range(3), LABELS)
        a_.set_yticks(range(3), LABELS)
        a_.set_xlabel("Predicted")
        a_.set_ylabel("Human label")
        a_.set_title(t)
        for i in range(3):
            for j in range(3):
                a_.text(j, i, format(m[i, j], f), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_data(clean_path, gold_path):
    for p in (clean_path, gold_path):
        if not Path(p).exists():
            raise SystemExit(f"File not found: {Path(p).resolve()}  (cwd = {Path.cwd()}) - upload it to Colab first.")
    corpus = pd.read_csv(clean_path)
    gold = pd.read_csv(gold_path)
    for c in ("review_id", "label"):
        if c not in gold.columns:
            raise SystemExit(f"{gold_path} lacks column {c!r}")
    gold = gold[["review_id", "label"]].dropna()
    gold["label"] = gold["label"].astype(str).str.strip().str.capitalize()
    bad = set(gold["label"]) - set(LABELS)
    if bad:
        raise SystemExit(f"unexpected labels in gold file: {bad} (expected {LABELS}); use the 3-class gold_labeled.csv")
    if gold["review_id"].duplicated().any():
        raise SystemExit("duplicated review_id in gold file")
    if corpus["text_seg"].isna().any():
        corpus["text_seg"] = corpus["text_seg"].fillna("")
    g = gold.merge(corpus, on="review_id", how="inner")
    if len(g) < len(gold):
        print(f"WARNING: {len(gold) - len(g)} gold review_ids are not in the corpus and were dropped")
    g["y"] = g["label"].map(L2I)
    return corpus, gold, g


def make_split(g, ratios, seed):
    strat = g["label"] + "|" + g["App"]
    strat = strat.where(strat.map(strat.value_counts()) >= 3, g["label"])          # avoid singleton strata
    tr, tmp = train_test_split(g, test_size=1 - ratios[0], stratify=strat, random_state=seed)
    s2 = strat.loc[tmp.index]
    s2 = s2.where(s2.map(s2.value_counts()) >= 2, tmp["label"])
    val, te = train_test_split(tmp, test_size=ratios[2] / (ratios[1] + ratios[2]), stratify=s2, random_state=seed)
    return tr, val, te


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="reviews_clean.csv")
    ap.add_argument("--gold", default="gold_labeled.csv")
    ap.add_argument("--model", default="vinai/phobert-base-v2")
    ap.add_argument("--outdir", default="phobert_outputs")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--weight-power", type=float, default=1.0, help="class weight = (N/(K*n_c))**power; 1 = balanced")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--split-seed", type=int, default=2024)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (training on CPU will be very slow)")

    corpus, gold, g = load_data(a.clean, a.gold)
    print("Gold labels:\n", g["label"].value_counts().to_string())
    tr, val, te = make_split(g, a.split, a.split_seed)
    pd.concat([tr.assign(split="train"), val.assign(split="val"), te.assign(split="test")])[
        ["review_id", "split", "label"]].to_csv(out / "splits.csv", index=False)
    print(f"train/val/test = {len(tr)}/{len(val)}/{len(te)}; Neutral in test = {(te['label'] == 'Neutral').sum()}")

    tok = AutoTokenizer.from_pretrained(a.model)
    lens = [len(tok.encode(t)) for t in g["text_seg"]]
    print(f"token length: median={np.median(lens):.0f} p95={np.percentile(lens, 95):.0f} "
          f"p99={np.percentile(lens, 99):.0f} max={max(lens)}  (max_len={a.max_len})")

    def enc(texts):
        return tok([str(t) for t in texts], truncation=True, max_length=a.max_len)

    tr_ds, val_ds, te_ds = (TextDS(enc(d["text_seg"]), d["y"].values) for d in (tr, val, te))
    all_ds = TextDS(enc(corpus["text_seg"]))
    cnt = np.bincount(tr["y"], minlength=3)
    weights = torch.tensor((len(tr) / (3 * np.maximum(cnt, 1))) ** a.weight_power, dtype=torch.float)
    print("train counts (Neg, Neu, Pos):", cnt, " class weights:", weights.numpy().round(2))

    # ---------------- baseline: TF-IDF + logistic regression ---------------- #
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True))
    base = make_pipeline(vec, LogisticRegression(max_iter=2000, class_weight="balanced", C=3))
    base.fit(tr["text_seg"], tr["y"])
    base_pred = base.predict(te["text_seg"])
    base_sc = scores(te["y"].values, base_pred)
    print("Baseline TF-IDF+LR (test):", {k: round(v, 4) for k, v in base_sc.items()})

    # ---------------- fine-tune, one run per seed ---------------- #
    val_f1, test_probs, corpus_probs = {}, [], []
    for seed in a.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)
        trainer = make_trainer(model, make_args(out / f"seed{seed}", a, seed), tr_ds, val_ds, tok, weights)
        trainer.train()
        val_f1[seed] = float(trainer.evaluate(val_ds)["eval_f1_macro"])
        test_probs.append(softmax(trainer.predict(te_ds).predictions))
        corpus_probs.append(softmax(trainer.predict(all_ds).predictions))
        sc = scores(te["y"].values, test_probs[-1].argmax(1))
        print(f"seed {seed}: val macro-F1={val_f1[seed]:.4f} | test acc={sc['accuracy']:.4f} "
              f"macro-F1={sc['macro_f1']:.4f} kappa={sc['cohen_kappa']:.4f}")
        del trainer, model
        torch.cuda.empty_cache()
        shutil.rmtree(out / f"seed{seed}", ignore_errors=True)         # free Colab disk (checkpoints)

    # ---------------- evaluation (test split, probability ensemble) ---------------- #
    y = te["y"].values
    per_seed = [scores(y, p.argmax(1)) for p in test_probs]
    ens_te = np.mean(test_probs, 0)
    ens_pred = ens_te.argmax(1)
    ens = scores(y, ens_pred)
    ci = bootstrap_ci(y, ens_pred)
    cm = confusion_matrix(y, ens_pred, labels=range(3))
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    cm_df.to_csv(out / "confusion_matrix_test.csv")
    summary = {
        "n_train_val_test": [len(tr), len(val), len(te)],
        "class_counts_train_neg_neu_pos": cnt.tolist(), "class_weights": weights.numpy().round(4).tolist(),
        "hyperparameters": {k: getattr(a, k.replace("-", "_")) for k in
                            ("lr", "bs", "epochs", "max_len", "weight_power", "seeds", "split_seed")},
        "val_macro_f1_by_seed": val_f1,
        "test_by_seed": dict(zip(map(str, a.seeds), per_seed)),
        "test_ensemble": ens, "test_ensemble_macro_f1_ci95": ci,
        "baseline_tfidf_lr_test": base_sc,
        "mcnemar_phobert_vs_tfidf": mcnemar_exact(y, ens_pred, base_pred),
        "library_versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"TEST SET RESULTS (n = {len(te)}, ensemble of {len(a.seeds)} seeds)")
    print(f"  Accuracy      : {ens['accuracy']:.4f}")
    print(f"  Macro-F1      : {ens['macro_f1']:.4f}   (bootstrap 95% CI {ci[0]:.3f} - {ci[1]:.3f})")
    print(f"  Cohen's kappa : {ens['cohen_kappa']:.4f}")
    print(f"  Per-seed macro-F1 mean +- sd: {np.mean([s['macro_f1'] for s in per_seed]):.4f} +- "
          f"{(np.std([s['macro_f1'] for s in per_seed], ddof=1) if len(per_seed) > 1 else 0):.4f}")
    print("\nConfusion matrix (rows = human label, columns = prediction):\n", cm_df.to_string())
    print("\n", classification_report(y, ens_pred, target_names=LABELS, zero_division=0, digits=3))
    print("=" * 62)
    pd.DataFrame(classification_report(y, ens_pred, target_names=LABELS, output_dict=True, zero_division=0)).T \
        .to_csv(out / "classification_report_test.csv")
    plot_cm(y, ens_pred, out / "confusion_matrix_test.png")
    err = te.assign(pred=[LABELS[i] for i in ens_pred], conf=ens_te.max(1))
    keep = [c for c in ("review_id", "App", "Platform", "Score", "text_raw", "label", "pred", "conf") if c in err.columns]
    err[err["pred"] != err["label"]][keep].sort_values("conf", ascending=False) \
        .to_csv(out / "errors_test.csv", index=False, encoding="utf-8-sig")

    # ---------------- predict the whole corpus ---------------- #
    P = np.mean(corpus_probs, 0)
    res = corpus.copy()                                     # the FULL reviews_clean.csv, all original columns kept
    res["pred_label"] = [LABELS[i] for i in P.argmax(1)]
    res["p_neg"], res["p_neu"], res["p_pos"] = P[:, 0], P[:, 1], P[:, 2]
    res["confidence"] = P.max(1)
    sp = pd.read_csv(out / "splits.csv").set_index("review_id")["split"]
    res["split"] = res["review_id"].map(sp).fillna("unlabelled")
    human = gold.set_index("review_id")["label"]
    res["sent_final"] = np.where(res["split"].isin(["train", "val"]), res["review_id"].map(human), res["pred_label"])
    res.to_csv(out / "reviews_phobert_predicted.csv", index=False, encoding="utf-8-sig")
    print(f"\nreviews_phobert_predicted.csv: {len(res)} rows")
    print("Predicted label distribution (all reviews):\n", res["pred_label"].value_counts().to_string())

    star = res["Score"].map({1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"})
    kw = cohen_kappa_score(star.map(L2I), res["pred_label"].map(L2I), weights="linear")
    print(f"weighted kappa (star-derived label vs model label, whole corpus) = {kw:.3f}")


if __name__ == "__main__":
    main()
