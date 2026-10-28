#!/usr/bin/env python3
"""
PHASE 4 (v2) - BERTopic on the PhoBERT-labelled corpus
=======================================================
Input   reviews_phobert_predicted.csv     (full corpus from 03_phobert_training_v2.py: all reviews_clean columns +
                                           pred_label, p_neg, p_neu, p_pos, confidence, split, sent_final)
Optional --gold-full gold_labeled_full.csv (from 02c: its label5 column marks the 78 reviews labelled Junk/Spam)

Topic corpus (printed step by step)
  1. is_topic_eligible : >= 4 words after cleaning and not a long copy-paste duplicate (re-derived from
                         n_words_clean / is_dup_longtext if the column is missing).
  2. Junk/Spam removed : any review whose label is "Junk/Spam" in the prediction file (label / label5 / sent_final /
                         pred_label, if such a value exists) OR in --gold-full (label5 == Junk/Spam). The 3-class PhoBERT
                         never predicts Junk/Spam, so the human Junk labels must come from --gold-full.

Two text versions are used ON PURPOSE
  embeddings  <- text_embed (natural, un-segmented text, what the multilingual encoder was trained on)
  c-TF-IDF    <- text_topic (word-segmented: 'đăng_nhập', 'tài_khoản' become meaningful keywords)

Model: SentenceTransformer paraphrase-multilingual-MiniLM-L12-v2 -> UMAP -> HDBSCAN -> c-TF-IDF (+ MMR keywords),
       Vietnamese stop-word list that KEEPS negations/failure words (không, chưa, bị, được, lỗi ...).

Outputs (outputs/bertopic/)
  reviews_with_topics.csv      the FULL input table + Topic (final, outliers re-assigned; <NA> if the review is not in the
                               topic corpus; -1 = still unassigned), topic_raw (HDBSCAN), topic_reassigned, topic_name,
                               in_topic_corpus  -> use this file for the regression phase
  topic_info.csv               id, size, name, top-10 c-TF-IDF keywords (+ MMR keywords)
  topic_keywords_long.csv      one row per (topic, rank, word, weight)
  topic_exemplars.csv          20 raw reviews closest to each topic centroid (naming / human validation)
  doc_topics.csv               review_id, topic_raw, topic, reassigned
  topic_by_app.csv, topic_by_platform.csv, topic_by_sentiment.csv, topic_by_app_sentiment_counts.csv
  intertopic_distance.html, barchart.html, hierarchy.html, heatmap.html (+ PNG if kaleido is installed)
  run_config.json              arguments, corpus sizes, topic count, outlier share, C_v / NPMI / diversity, library versions
  model/                       saved BERTopic model;  grid_results.csv with --grid

Colab:
  !pip install -q bertopic sentence-transformers umap-learn hdbscan gensim kaleido pandas scikit-learn
  !python 04_bertopic_modeling_v2.py --pred reviews_phobert_predicted.csv --gold-full gold_labeled_full.csv
  (optional) --nr-topics 20 to merge after inspecting the first run;  --grid for the sensitivity grid
"""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Vietnamese function-word stop list. Deliberately KEEPS negations and failure words (không, chưa, chẳng, bị, được,
# lỗi ...): scikit-learn removes stop words BEFORE building bigrams, so removing 'không' would destroy
# 'không đăng_nhập'. Brand names and placeholder tokens are removed so that topics are not "about Cake".
# Extend with --stopwords-file (one word per line) after reading the first topic_info.csv.
# --------------------------------------------------------------------------- #
VN_STOPWORDS = sorted(set("""
và là của có cho thì mà này đó nó tôi mình em anh chị bạn ạ nhé nhỉ nha à ơi ừ vậy thế
rất quá cũng đã đang sẽ vì nên để với các những một mọi khi lại ra vào lên xuống rồi nhưng nếu hay hoặc
cái con người ai gì nào đây kia đi về tại trong ngoài trên dưới sau trước còn vẫn cứ luôn thôi
thấy biết làm dùng xài sử_dụng ok
app ứng_dụng ngân_hàng bank ngân hàng ứng dụng
timo cake vikki vpbank bvbank hdbank dongabank donga viet_capital be_group
phone email url number emojipos emojineg emojineu
""".split()))

JUNK_LABELS = {"junk/spam", "junk", "spam"}
LABEL_COLS = ("label", "label5", "sent_final", "pred_label")


# --------------------------------------------------------------------------- #
# Corpus preparation (pure pandas -> easy to test)
# --------------------------------------------------------------------------- #
def prepare_corpus(df, gold_full=None, min_words=4, verbose=True):
    """Return (mask_in_topic_corpus, report). `df` is the full prediction table."""
    n = len(df)
    if "is_topic_eligible" in df.columns:
        elig = df["is_topic_eligible"].astype(str).str.lower().isin(["true", "1", "1.0"])
    else:
        elig = (df["n_words_clean"] >= min_words) & (~df.get("is_dup_longtext", False).astype(bool))
    junk = pd.Series(False, index=df.index)
    for c in LABEL_COLS:
        if c in df.columns:
            junk |= df[c].astype(str).str.strip().str.lower().isin(JUNK_LABELS)
    n_junk_in_file = int(junk.sum())
    if gold_full is not None:
        g = pd.read_csv(gold_full)
        col = "label5" if "label5" in g.columns else "label"
        ids = set(g.loc[g[col].astype(str).str.strip().str.lower().isin(JUNK_LABELS), "review_id"])
        junk |= df["review_id"].isin(ids)
    mask = elig & ~junk
    rep = {"n_reviews": int(n), "n_topic_eligible": int(elig.sum()), "n_junk_marked": int(junk.sum()),
           "n_junk_marked_in_file": n_junk_in_file, "n_junk_among_eligible": int((elig & junk).sum()),
           "n_topic_corpus": int(mask.sum())}
    if verbose:
        print(f"Reviews in file                    : {rep['n_reviews']}")
        print(f"  topic-eligible (>= {min_words} words, no dup): {rep['n_topic_eligible']}")
        print(f"  Junk/Spam marked                  : {rep['n_junk_marked']} (of which eligible: {rep['n_junk_among_eligible']})")
        print(f"  TOPIC CORPUS                      : {rep['n_topic_corpus']}")
        if rep["n_junk_marked"] == 0:
            print("  [warn] no Junk/Spam review was found; pass --gold-full gold_labeled_full.csv to exclude the human-labelled junk")
    return mask.values, rep


def load_stopwords(path=None):
    sw = set(VN_STOPWORDS)
    if path:
        sw |= {w.strip() for w in Path(path).read_text(encoding="utf-8").split() if w.strip()}
    return sorted(sw)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(a, seed, docs, stopwords):
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_m = UMAP(n_neighbors=a.n_neighbors, n_components=a.n_components, min_dist=0.0, metric="cosine",
                  random_state=seed)
    hdb = HDBSCAN(min_cluster_size=a.min_cluster_size, min_samples=a.min_samples, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    # BERTopic fits the vectoriser on ONE concatenated document PER TOPIC, so min_df inside BERTopic would count topics.
    # The vocabulary is therefore fixed on the REVIEW level first (min_df = number of reviews).
    kw = dict(ngram_range=(1, 2), stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")
    vocab = CountVectorizer(min_df=a.min_df, **kw).fit(docs).get_feature_names_out().tolist()
    vec = CountVectorizer(vocabulary=vocab, **kw)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    return umap_m, hdb, vec, ctfidf


def fit_bertopic(docs, E, a, seed, stopwords, embedder=None, mmr=True):
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance

    umap_m, hdb, vec, ctfidf = build_models(a, seed, docs, stopwords)
    rep = {"MMR": MaximalMarginalRelevance(diversity=a.mmr_diversity)} if (mmr and embedder is not None) else None
    # language="multilingual" is ESSENTIAL: the default silently deletes non-ASCII characters (Vietnamese diacritics, '_').
    tm = BERTopic(language="multilingual", embedding_model=embedder, umap_model=umap_m, hdbscan_model=hdb,
                  vectorizer_model=vec, ctfidf_model=ctfidf, representation_model=rep, top_n_words=10,
                  calculate_probabilities=False, verbose=True)
    topics, _ = tm.fit_transform(docs, embeddings=E)
    return tm, np.array(topics)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def topic_words(tm, n=10):
    return {t: [w for w, _ in tm.get_topic(t)[:n]] for t in tm.get_topics() if t != -1}


def topic_diversity(tw):
    words = [w for ws in tw.values() for w in ws]
    return len(set(words)) / max(len(words), 1)


def coherence(tw, token_docs):
    """C_v and NPMI (gensim). Documents = unigrams + adjacent bigrams so bigram keywords exist in the dictionary."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    texts = [t + [f"{x} {y}" for x, y in zip(t, t[1:])] for t in token_docs]
    d = Dictionary(texts)
    topics = [[w for w in ws if w in d.token2id] for ws in tw.values()]
    topics = [t for t in topics if len(t) >= 3]
    return {m: CoherenceModel(topics=topics, texts=texts, dictionary=d, coherence=m, processes=1).get_coherence()
            for m in ("c_v", "c_npmi")}


def jaccard_stability(tw_a, tw_b):
    scores = []
    for wa in tw_a.values():
        sa = set(wa)
        scores.append(max((len(sa & set(wb)) / len(sa | set(wb)) for wb in tw_b.values()), default=0))
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="reviews_phobert_predicted.csv")
    ap.add_argument("--gold-full", default=None, help="gold_labeled_full.csv (label5 == Junk/Spam rows are excluded)")
    ap.add_argument("--outdir", default="outputs/bertopic")
    ap.add_argument("--embedder", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--min-cluster-size", type=int, default=50)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--mmr-diversity", type=float, default=0.3)
    ap.add_argument("--nr-topics", type=int, default=None, help="merge topics hierarchically to this number")
    ap.add_argument("--outlier-threshold", type=float, default=0.30,
                    help="cosine similarity needed to move an outlier into its nearest topic (0 = move all)")
    ap.add_argument("--stopwords-file", default=None, help="extra stop words, whitespace separated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(a.pred).exists():
        raise SystemExit(f"File not found: {Path(a.pred).resolve()} - upload reviews_phobert_predicted.csv to Colab first.")

    full = pd.read_csv(a.pred)
    for c in ("review_id", "text_embed", "text_topic", "text_raw", "App", "Score"):
        if c not in full.columns:
            raise SystemExit(f"{a.pred} lacks column {c!r}")
    mask, rep = prepare_corpus(full, a.gold_full, a.min_words)
    df = full[mask].reset_index(drop=True)
    docs_embed = df["text_embed"].fillna("").tolist()
    docs = df["text_topic"].fillna("").tolist()
    token_docs = [d.split() for d in docs]
    stopwords = load_stopwords(a.stopwords_file)
    print(f"Stop words: {len(stopwords)} (negations and failure words kept)")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(a.embedder)
    tag = hashlib.md5("\n".join(docs_embed).encode("utf-8")).hexdigest()[:8]        # cache tied to the exact texts
    cache = out / f"embeddings_{Path(a.embedder).name}_{len(df)}_{tag}.npy"
    if cache.exists():
        E = np.load(cache)
    else:
        E = embedder.encode(docs_embed, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        np.save(cache, E)

    # ------------------------------------------------------------------ grid ------------------------------ #
    if a.grid:
        rows, runs = [], {}
        for nn, mcs in itertools.product([10, 15, 30], [30, 50, 80]):
            for seed in (42, 43, 44):
                a2 = argparse.Namespace(**{**vars(a), "n_neighbors": nn, "min_cluster_size": mcs})
                tm, tp = fit_bertopic(docs, E, a2, seed, stopwords, embedder=None, mmr=False)
                tw = topic_words(tm)
                runs[(nn, mcs, seed)] = tw
                if len(tw) >= 3:
                    c = coherence(tw, token_docs)
                    rows.append(dict(n_neighbors=nn, min_cluster_size=mcs, seed=seed, n_topics=len(tw),
                                     outlier_share=float((tp == -1).mean()), c_v=c["c_v"], c_npmi=c["c_npmi"],
                                     diversity=topic_diversity(tw)))
        res = pd.DataFrame(rows)
        stab = [dict(n_neighbors=nn, min_cluster_size=mcs, seed=seed,
                     jaccard_vs_seed42=jaccard_stability(runs[(nn, mcs, 42)], tw) if seed != 42 else np.nan)
                for (nn, mcs, seed), tw in runs.items()]
        res = res.merge(pd.DataFrame(stab), on=["n_neighbors", "min_cluster_size", "seed"])
        res.to_csv(out / "grid_results.csv", index=False)
        print(res.groupby(["n_neighbors", "min_cluster_size"]).mean(numeric_only=True).round(3).to_string())
        return

    # ------------------------------------------------------------------ main run -------------------------- #
    tm, topics = fit_bertopic(docs, E, a, a.seed, stopwords, embedder=embedder)
    n_raw = len(set(topics)) - (1 if -1 in topics else 0)
    raw_outliers = float((topics == -1).mean())
    print(f"HDBSCAN: {n_raw} topics, outliers = {raw_outliers:.1%}")
    topic_raw = topics.copy()
    if a.nr_topics:
        tm.reduce_topics(docs, nr_topics=a.nr_topics)
        topics = np.array(tm.topics_)

    # re-assign outliers by embedding similarity, then recompute keywords on the final assignment
    new_topics = np.array(tm.reduce_outliers(docs, topics.tolist(), strategy="embeddings", embeddings=E,
                                             threshold=a.outlier_threshold))
    _, _, vec, ctfidf = build_models(a, a.seed, docs, stopwords)
    tm.update_topics(docs, topics=new_topics.tolist(), vectorizer_model=vec, ctfidf_model=ctfidf)
    unassigned = float((new_topics == -1).mean())
    print(f"After reassignment: unassigned = {unassigned:.1%} (threshold {a.outlier_threshold}) "
          f"-> topic coverage {1 - unassigned:.1%}")

    # ------------------------------------------------------------------ tables ---------------------------- #
    info = tm.get_topic_info()
    tw = topic_words(tm)
    info["top_keywords"] = info["Topic"].map(lambda t: ", ".join(tw.get(t, [])))
    if "MMR" in info.columns:
        info["mmr_keywords"] = info["MMR"].map(lambda v: ", ".join(v) if isinstance(v, list) else v)
    info.drop(columns=[c for c in ("Representation", "Representative_Docs", "MMR") if c in info.columns]) \
        .to_csv(out / "topic_info.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(topic=t, rank=i + 1, word=w, ctfidf=s) for t in tm.get_topics() if t != -1
                  for i, (w, s) in enumerate(tm.get_topic(t)[:10])]) \
        .to_csv(out / "topic_keywords_long.csv", index=False, encoding="utf-8-sig")

    ex = []
    for t in sorted(set(new_topics) - {-1}):
        idx = np.where(new_topics == t)[0]
        cen = E[idx].mean(0); cen /= np.linalg.norm(cen)
        top = idx[np.argsort(-(E[idx] @ cen))[:20]]
        ex += [dict(topic=t, rank=r + 1, review_id=int(df.loc[i, "review_id"]), App=df.loc[i, "App"],
                    Score=int(df.loc[i, "Score"]), text_raw=df.loc[i, "text_raw"]) for r, i in enumerate(top)]
    pd.DataFrame(ex).to_csv(out / "topic_exemplars.csv", index=False, encoding="utf-8-sig")

    dt = pd.DataFrame({"review_id": df["review_id"], "topic_raw": topic_raw, "topic": new_topics})
    dt["reassigned"] = dt["topic_raw"] != dt["topic"]
    dt.to_csv(out / "doc_topics.csv", index=False)

    # ---- append the assigned Topic to the FULL dataset (for the regression phase) ----
    names = {r.Topic: r.Name for r in tm.get_topic_info().itertuples()}
    app = full.merge(dt.rename(columns={"topic": "Topic", "reassigned": "topic_reassigned"}), on="review_id", how="left")
    app["Topic"] = app["Topic"].astype("Int64")
    app["topic_raw"] = app["topic_raw"].astype("Int64")
    app["topic_name"] = app["Topic"].map(lambda t: names.get(int(t)) if pd.notna(t) else None)
    app["in_topic_corpus"] = app["Topic"].notna()
    assert len(app) == len(full) and app["in_topic_corpus"].sum() == len(df)
    app.to_csv(out / "reviews_with_topics.csv", index=False, encoding="utf-8-sig")
    print(f"reviews_with_topics.csv: {len(app)} rows, {int(app['in_topic_corpus'].sum())} with a Topic")

    # ---- cross-tabs ----
    m = app[app["in_topic_corpus"]]
    pd.crosstab(m["Topic"], m["App"], normalize="columns").round(4).to_csv(out / "topic_by_app.csv")
    pd.crosstab(m["Topic"], m["Platform"], normalize="columns").round(4).to_csv(out / "topic_by_platform.csv")
    for col, fn in (("pred_label", "topic_by_sentiment.csv"),):
        if col in m:
            pd.crosstab(m["Topic"], m[col]).to_csv(out / fn)
    scol = "sent_final" if "sent_final" in m else "pred_label"
    if scol in m:
        pd.crosstab([m["Topic"], m[scol]], m["App"]).to_csv(out / "topic_by_app_sentiment_counts.csv")

    # ------------------------------------------------------------------ evaluation + visuals -------------- #
    cfg = {"args": vars(a), "corpus": rep, "n_topics_hdbscan": n_raw, "outlier_share_hdbscan": raw_outliers,
           "n_topics_final": len(tw), "unassigned_after_reassignment": unassigned, "stopwords": len(stopwords)}
    try:
        c = coherence(tw, token_docs)
        cfg.update(c_v=c["c_v"], c_npmi=c["c_npmi"], diversity=topic_diversity(tw))
        print(f"C_v = {c['c_v']:.3f} | NPMI = {c['c_npmi']:.3f} | diversity = {topic_diversity(tw):.3f}")
    except Exception as e:
        print("[warn] coherence skipped:", e)
    import importlib.metadata as md
    cfg["versions"] = {p: md.version(p) for p in ("bertopic", "sentence-transformers", "umap-learn", "hdbscan", "gensim")
                       if p in {d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]}}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    try:
        tm.save(str(out / "model"), serialization="safetensors", save_ctfidf=True, save_embedding_model=a.embedder)
    except Exception as e:
        print("[warn] model not saved:", e)
    figs = {"intertopic_distance": lambda: tm.visualize_topics(),
            "barchart": lambda: tm.visualize_barchart(top_n_topics=min(20, len(tw)), n_words=8),
            "hierarchy": lambda: tm.visualize_hierarchy(),
            "heatmap": lambda: tm.visualize_heatmap()}
    for name, make in figs.items():
        try:
            fig = make()
            fig.write_html(out / f"{name}.html")
            try:
                fig.write_image(out / f"{name}.png", scale=2)
            except Exception:
                pass
        except Exception as e:
            print(f"[warn] {name} skipped: {e}")
    print("Saved to", out)


if __name__ == "__main__":
    main()
