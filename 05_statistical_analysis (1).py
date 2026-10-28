#!/usr/bin/env python3
"""
PHASE 5 - Statistical analysis for H1, H2, H3, H4a, H4b, H5
============================================================
Inputs
  --data      reviews_with_topics.csv        (from 04_bertopic_modeling_v2.py: Topic, topic_raw, sent_final, pred_label ...)
  --codebook  codebook_topic_mapping.csv     (YOUR FINAL codebook; columns: Topic, topic_name_en, dimension, alt_dimension)
  --gold-full gold_labeled_full.csv          (optional; only used to repeat H1/H2 without the 78 Junk/Spam reviews)

Valid `dimension` values: Efficiency, System Availability, Fulfillment, Privacy, Responsiveness, Compensation, Contact,
Product_Other, General_Evaluation, EXCLUDED (topic that groups texts by writing style), UNASSIGNED (Topic = -1).
`alt_dimension` (optional, may be empty) is the alternative reading used ONLY in the H4a sensitivity analysis.
`brand_mention_cluster` (optional; 'Yes' marks topics clustered around brand names) is used ONLY for a robustness rerun of H3 and H4a
without those topics. Codebook version: BERTopic run 2 (33 topics, topics 0-32; Topic -1 = 2 unassigned reviews).

Design (Section 3.7)
  H1   chi-square App x sentiment (+ Cramer's V, adjusted standardized residuals, Holm-adjusted pairwise tests);
       multinomial logit with Platform, quarter and log(words+1) controls, standard errors clustered by App x month;
       LR test and cluster-robust Wald test of the App effect; adjusted predicted probabilities; stars: Kruskal-Wallis + Dunn.
  H2   multinomial logit with App x Platform: LR test and cluster-robust Wald test of the interaction, on the full period and
       in the common window (from 2025-03-01); per-app platform contrasts with Holm.
  H3   chi-square Topic x App among Negative reviews (permutation p, 2,000 permutations), Cramer's V, adjusted residuals;
       repeated in the common window and on the unreduced HDBSCAN topics; dimension x App among Negative reviews.
  H4a  ordered logit rating ~ dimension (reference Product_Other) + App + Platform + quarter + log length, cluster-robust SE;
       odds ratios, Holm-adjusted pairwise contrasts, predicted probabilities, proportional-odds check, high-vs-low binary
       logit, VIF, and a sensitivity analysis that moves one topic at a time to its alt_dimension.
  H4b  ordered logit with dimension x polarity indicators (reference: Neutral); Wald test b_neg + b_pos = 0; asymmetry index.
  H5   net sentiment (% Positive - % Negative) by dimension x App; percentile bootstrap (5,000) of pairwise app differences
       where both cells have >= 30 reviews; Holm adjustment.

Colab:
  !pip install -q statsmodels scipy pandas numpy scikit-learn
  !python 05_statistical_analysis.py --data reviews_with_topics.csv --codebook codebook_topic_mapping.csv \
        --gold-full gold_labeled_full.csv
Outputs: outputs/stats/*.csv, stats_log.txt (everything printed), analysis_config.json, and outputs_stats.zip.
"""
import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SENT_CODE = {"Negative": 0, "Neutral": 1, "Positive": 2}
SENT_ORDER = ["Negative", "Neutral", "Positive"]
DIMENSIONS = ["Efficiency", "System Availability", "Fulfillment", "Privacy", "Responsiveness", "Compensation",
              "Contact", "Product_Other", "General_Evaluation"]
NON_ANALYTIC = {"EXCLUDED", "UNASSIGNED"}
REF_DIM = "Product_Other"
OUT = Path("outputs/stats")


# ----------------------------------------------------------------------------------------------------------------- #
# utilities
# ----------------------------------------------------------------------------------------------------------------- #
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s)

    def flush(self):
        self.o.flush(); self.f.flush()


def save(df, name, index=False, show=True, title=None):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=index, encoding="utf-8-sig")
    if show:
        print(f"\n--- {title or name} -> {name}")
        print(df.round(4).to_string(index=index))


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(n)
    prev = 1.0
    for k, i in enumerate(order):
        rank = n - k
        prev = min(prev, p[i] * n / rank)
        adj[i] = prev
    return adj


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else np.nan


def adj_resid(ct):
    o = ct.values.astype(float)
    n = o.sum()
    r = o.sum(1, keepdims=True)
    c = o.sum(0, keepdims=True)
    e = r @ c / n
    z = (o - e) / np.sqrt(e * (1 - r / n) * (1 - c / n))
    return pd.DataFrame(z, index=ct.index, columns=ct.columns)


def chi_block(df, row, col, label, perm=None, rng=None):
    """chi-square of row x col with V; optional permutation p (shuffle the `col` labels)."""
    ct = pd.crosstab(df[row], df[col])
    ct = ct.loc[ct.sum(1) > 0, ct.sum(0) > 0]
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    res = dict(test=label, n=int(ct.values.sum()), chi2=chi2, df=dof, p_asymptotic=p, cramers_v=cramers_v(ct),
               pct_cells_expected_lt5=float((exp < 5).mean() * 100), min_expected=float(exp.min()))
    if perm:
        x = df[row].astype("category").cat.codes.values
        y = df[col].astype("category").cat.codes.values.copy()
        nx, ny = x.max() + 1, y.max() + 1
        def chi_of(yv):
            t = np.bincount(x * ny + yv, minlength=nx * ny).reshape(nx, ny).astype(float)
            t = t[t.sum(1) > 0][:, t.sum(0) > 0]
            e = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / t.sum()
            return ((t - e) ** 2 / e).sum()
        obs = chi_of(y)
        ge = 0
        for _ in range(perm):
            ge += chi_of(rng.permutation(y)) >= obs - 1e-9
        res["p_permutation"] = (ge + 1) / (perm + 1)
    return ct, res


def prep_design(df, extra_cols=()):
    """Control variables shared by every model."""
    d = df.copy()
    d["log_len"] = np.log1p(d["n_words_clean"].astype(float))
    d["cluster"] = pd.factorize(d["App"].astype(str) + "|" + d["year_month"].astype(str))[0]
    d["quarter"] = d["quarter"].astype(str)
    return d


def controls_matrix(d, with_app=True, with_platform=True, drop_first=True):
    parts = []
    if with_app:
        parts.append(pd.get_dummies(d["App"], prefix="App", drop_first=drop_first))
    if with_platform:
        parts.append(pd.get_dummies(d["Platform"], prefix="Platform", drop_first=drop_first))
    parts.append(pd.get_dummies(d["quarter"], prefix="Q", drop_first=drop_first))
    parts.append(d[["log_len"]])
    return pd.concat(parts, axis=1).astype(float)


def drop_constant(X):
    keep = [c for c in X.columns if X[c].nunique() > 1]
    return X[keep]


# ----------------------------------------------------------------------------------------------------------------- #
# multinomial logit helpers (H1, H2)
# ----------------------------------------------------------------------------------------------------------------- #
def fit_mnl(d, X, cluster=True):
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.MNLogit(d["y"].values, Xc)
    if cluster:
        return m.fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d["cluster"].values})
    return m.fit(disp=0, maxiter=300)


def mnl_wald(res, predicate):
    """Cluster-robust Wald chi2 for all coefficients (all equations) whose name satisfies predicate."""
    cv = res.cov_params()
    sel = [i for i, (eq, nm) in enumerate(cv.index) if predicate(nm)]
    beta = np.array([res.params.iloc[:, int(cv.index[i][0]) - 1][cv.index[i][1]] for i in sel])
    V = cv.values[np.ix_(sel, sel)]
    w = float(beta @ np.linalg.pinv(V) @ beta)
    return w, len(sel), float(stats.chi2.sf(w, len(sel)))


def mnl_block(d, label, spec):
    """spec = 'app' (H1) or 'interaction' (H2). Returns a dict with LR and Wald tests."""
    import statsmodels.api as sm
    base = controls_matrix(d, with_app=True, with_platform=True)
    if spec == "app":
        X_full = base
        X_red = base[[c for c in base.columns if not c.startswith("App_")]]
        pred = lambda nm: nm.startswith("App_")
    else:
        apps = [c for c in base.columns if c.startswith("App_")]
        plats = [c for c in base.columns if c.startswith("Platform_")]
        inter = pd.DataFrame({f"{a}:{p}": base[a] * base[p] for a in apps for p in plats})
        X_full = pd.concat([base, inter], axis=1)
        X_red = base
        pred = lambda nm: ":" in nm
    X_full, X_red = drop_constant(X_full), drop_constant(X_red)
    f_plain, r_plain = fit_mnl(d, X_full, cluster=False), fit_mnl(d, X_red, cluster=False)
    lr = 2 * (f_plain.llf - r_plain.llf)
    dof = X_full.shape[1] - X_red.shape[1]
    f_cl = fit_mnl(d, X_full, cluster=True)
    w, wdf, wp = mnl_wald(f_cl, pred)
    return dict(test=label, n=len(d), lr_chi2=lr, lr_df=dof * 2, lr_p=float(stats.chi2.sf(lr, dof * 2)),
                wald_chi2_cluster=w, wald_df=wdf, wald_p_cluster=wp, n_clusters=int(d["cluster"].nunique())), f_cl, X_full


def adjusted_probs(res, X, d, var_prefix, levels):
    """Average adjusted predicted probabilities when every review is set to each level of an App/Platform factor."""
    import statsmodels.api as sm
    rows = []
    for lv in levels:
        Z = X.copy()
        for c in [c for c in Z.columns if c.startswith(var_prefix + "_") and ":" not in c]:
            Z[c] = 1.0 if c == f"{var_prefix}_{lv}" else 0.0
        if True:      # keep interaction columns consistent
            for c in [c for c in Z.columns if ":" in c]:
                a, p = c.split(":")
                Z[c] = Z[a] * Z[p]
        P = np.asarray(res.predict(sm.add_constant(Z, has_constant="add")))
        rows.append(dict(level=lv, **{f"P({s})": float(np.mean(P[:, k])) for s, k in SENT_CODE.items()}))
    return pd.DataFrame(rows)


def dunn(df, group, value):
    g = df[group]
    r = stats.rankdata(df[value])
    n = len(df)
    levels = sorted(g.unique())
    ranks = {l: r[(g == l).values] for l in levels}
    ties = stats.tiecorrect(r)
    out = []
    for a, b in combinations(levels, 2):
        na, nb = len(ranks[a]), len(ranks[b])
        se = np.sqrt((n * (n + 1) / 12.0) * ties * (1 / na + 1 / nb))
        z = (ranks[a].mean() - ranks[b].mean()) / se
        out.append(dict(group_a=a, group_b=b, z=z, p=2 * stats.norm.sf(abs(z))))
    o = pd.DataFrame(out)
    o["p_holm"] = holm(o["p"])
    return o


# ----------------------------------------------------------------------------------------------------------------- #
# ordered logit helpers (H4a, H4b)
# ----------------------------------------------------------------------------------------------------------------- #
def fit_ordered(y, X, groups):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    m = OrderedModel(y, X, distr="logit")
    best = None
    for meth in ("bfgs", "lbfgs", "nm"):
        try:
            r = m.fit(method=meth, disp=0, maxiter=2000, cov_type="cluster", cov_kwds={"groups": groups})
            if np.isfinite(r.llf):
                return r
        except Exception as e:
            best = e
    raise RuntimeError(f"ordered logit failed: {best}")


def coef_table(res, names):
    p = res.params
    ci = res.conf_int()
    t = pd.DataFrame({"term": names, "coef": p[names].values, "se": res.bse[names].values,
                      "z": res.tvalues[names].values, "p": res.pvalues[names].values,
                      "OR": np.exp(p[names].values), "OR_lo": np.exp(ci.loc[names, 0].values),
                      "OR_hi": np.exp(ci.loc[names, 1].values)})
    return t


def wald_linear(res, names, weights):
    """Wald chi2(1) test for sum(weights*coef) = 0 using the cluster-robust covariance."""
    cov = res.cov_params().loc[names, names].values
    w = np.asarray(weights, float)
    est = float(w @ res.params[names].values)
    se = float(np.sqrt(w @ cov @ w))
    z = est / se if se > 0 else np.nan
    return est, se, 2 * stats.norm.sf(abs(z))


def dimension_design(d, dims_present, ref=REF_DIM):
    D = pd.DataFrame({f"dim_{k}": (d["dimension"] == k).astype(float) for k in dims_present if k != ref}, index=d.index)
    return D


def run_h4a(d, label, dims=None, verbose=False):
    d = d.copy()
    counts = d["dimension"].value_counts()
    dims_present = [k for k in DIMENSIONS if counts.get(k, 0) >= 30]
    d = d[d["dimension"].isin(dims_present)]
    D = dimension_design(d, dims_present)
    C = controls_matrix(d)
    X = drop_constant(pd.concat([D, C], axis=1))
    y = d["Score"].astype(int).values
    res = fit_ordered(pd.Series(y, index=X.index), X, d["cluster"].values)
    names = list(D.columns)
    tab = coef_table(res, names)
    tab.insert(0, "model", label)
    tab["p_holm_vs_ref"] = holm(tab["p"].values)
    tab["n"] = len(d)
    return d, X, res, tab, dims_present


# ----------------------------------------------------------------------------------------------------------------- #
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reviews_with_topics.csv")
    ap.add_argument("--codebook", default="codebook_topic_mapping.csv")
    ap.add_argument("--gold-full", default=None)
    ap.add_argument("--sentiment-col", default="sent_final", choices=["sent_final", "pred_label"])
    ap.add_argument("--outdir", default="outputs/stats")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--min-cell", type=int, default=30, help="minimum reviews per app x dimension cell for H5")
    ap.add_argument("--skip-sensitivity", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUT / "stats_log.txt")
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)

    # ------------------------------------------------------------------ data ------------------------------------ #
    raw = pd.read_csv(a.data)
    cb = pd.read_csv(a.codebook)
    cb["dimension"] = cb["dimension"].astype(str).str.strip()
    bad = set(cb["dimension"]) - set(DIMENSIONS) - NON_ANALYTIC
    if bad:
        raise SystemExit(f"Unknown dimension labels in the codebook: {bad}")
    if "alt_dimension" not in cb:
        cb["alt_dimension"] = ""
    cb["alt_dimension"] = cb["alt_dimension"].fillna("").astype(str).str.strip()
    print(f"Reviews: {len(raw)} | codebook topics: {len(cb)}")
    raw["sent"] = raw[a.sentiment_col].astype(str)
    assert set(raw["sent"]) <= set(SENT_ORDER), set(raw["sent"])
    raw["y"] = raw["sent"].map(SENT_CODE)
    raw["is_junk"] = False
    if a.gold_full:
        g = pd.read_csv(a.gold_full)
        col = "label5" if "label5" in g.columns else "label"
        junk_ids = set(g.loc[g[col].astype(str).str.lower().str.contains("junk|spam"), "review_id"])
        raw["is_junk"] = raw["review_id"].isin(junk_ids)
        print(f"Junk/Spam reviews flagged from --gold-full: {int(raw['is_junk'].sum())}")
    df = prep_design(raw)
    # dimension of the final topic (topic_raw -1 -> unassigned) and of the unreduced HDBSCAN topic
    m = dict(zip(cb["Topic"], cb["dimension"]))
    df["dimension"] = df["Topic"].map(lambda t: m.get(int(t)) if pd.notna(t) else np.nan)
    df["dimension_raw"] = df["topic_raw"].map(lambda t: m.get(int(t)) if pd.notna(t) else np.nan)
    df["in_topic_corpus"] = df["in_topic_corpus"].astype(bool)
    an = df[df["in_topic_corpus"] & df["dimension"].isin(DIMENSIONS)].copy()          # analysable topic reviews
    print(f"Topic corpus: {int(df['in_topic_corpus'].sum())} | assigned to a service-quality dimension / residual category: "
          f"{len(an)} ({len(an) / df['in_topic_corpus'].sum():.1%}) | excluded-topic reviews: "
          f"{int((df['dimension'] == 'EXCLUDED').sum())} | unassigned: {int((df['dimension'] == 'UNASSIGNED').sum())}")
    cfg = dict(args=vars(a), n_reviews=len(df), n_topic_corpus=int(df["in_topic_corpus"].sum()), n_dimension_sample=len(an))

    # ------------------------------------------------------------------ descriptives ---------------------------- #
    desc = (df.groupby("App").agg(n=("review_id", "size"), mean_stars=("Score", "mean"),
                                  pct_neg=("sent", lambda s: (s == "Negative").mean() * 100),
                                  pct_neu=("sent", lambda s: (s == "Neutral").mean() * 100),
                                  pct_pos=("sent", lambda s: (s == "Positive").mean() * 100)).reset_index())
    save(desc, "00_descriptives_by_app.csv", title="Descriptives by app")
    d2 = (df.groupby(["App", "Platform"]).agg(n=("review_id", "size"), mean_stars=("Score", "mean"),
                                              pct_neg=("sent", lambda s: (s == "Negative").mean() * 100),
                                              pct_pos=("sent", lambda s: (s == "Positive").mean() * 100)).reset_index())
    save(d2, "00_descriptives_by_app_platform.csv", title="Descriptives by app x platform")
    dsz = (an.groupby("dimension").agg(n=("review_id", "size"), pct_of_analysed=("review_id", lambda s: len(s) / len(an) * 100),
                                       mean_stars=("Score", "mean"), pct_1star=("Score", lambda s: (s == 1).mean() * 100),
                                       pct_neg=("sent", lambda s: (s == "Negative").mean() * 100),
                                       pct_pos=("sent", lambda s: (s == "Positive").mean() * 100)).reset_index())
    dsz = dsz.sort_values("n", ascending=False)
    save(dsz, "00_dimension_summary.csv", title="Dimension summary (reviews with a topic)")
    save(pd.crosstab(an["dimension"], an["App"], margins=True), "00_dimension_by_app_counts.csv", index=True,
         title="Dimension x App counts")
    tsum = (df[df["in_topic_corpus"]].groupby("Topic").agg(n=("review_id", "size"), mean_stars=("Score", "mean"),
            pct_neg=("sent", lambda s: (s == "Negative").mean() * 100)).reset_index()
            .merge(cb[["Topic", "topic_name_en", "dimension"]], on="Topic", how="left"))
    save(tsum, "00_topic_summary_with_codebook.csv", title="Topic summary (Table 4.12 source)")

    # ------------------------------------------------------------------ H1 -------------------------------------- #
    print("\n================ H1: sentiment by app ================")
    try:
        ct, r1 = chi_block(df, "App", "sent", "H1 chi-square App x sentiment (all reviews)")
        ct = ct[[s for s in SENT_ORDER if s in ct.columns]]
        save(ct.reset_index(), "H1_crosstab_counts.csv", title="H1 counts")
        save((ct.div(ct.sum(1), axis=0) * 100).reset_index(), "H1_crosstab_row_pct.csv", title="H1 row %")
        save(adj_resid(ct).reset_index(), "H1_adjusted_residuals.csv", title="H1 adjusted standardized residuals")
        pw = []
        for x, y_ in combinations(sorted(df["App"].unique()), 2):
            _, r = chi_block(df[df["App"].isin([x, y_])], "App", "sent", f"{x} vs {y_}")
            pw.append(r)
        pw = pd.DataFrame(pw)
        pw["p_holm"] = holm(pw["p_asymptotic"])
        save(pd.concat([pd.DataFrame([r1]), pw]), "H1_chi_square.csv", title="H1 chi-square and pairwise tests")
        res_rows, probs = [], []
        for lab, sub in (("H1 multinomial logit, full period", df),
                         ("H1 multinomial logit, common window", df[df["in_common_window"].astype(bool)])):
            row, fit, X = mnl_block(sub, lab, "app")
            res_rows.append(row)
            probs.append(adjusted_probs(fit, X, sub, "App", sorted(sub["App"].unique())).assign(model=lab))
            if lab.endswith("full period"):
                full_fit = fit
                pr = full_fit.params.copy(); pr.columns = ["Neutral_vs_Negative", "Positive_vs_Negative"]
                se = full_fit.bse.copy(); se.columns = pr.columns
                pv = full_fit.pvalues.copy(); pv.columns = pr.columns
                save(pd.concat({"coef": pr, "se": se, "p": pv}, axis=1).reset_index().rename(columns={"index": "term"}),
                     "H1_mnlogit_coefficients_full.csv", show=False)
        save(pd.DataFrame(res_rows), "H1_mnlogit_tests.csv", title="H1 App effect (LR and cluster-robust Wald)")
        save(pd.concat(probs), "H1_adjusted_probabilities.csv", title="H1 adjusted predicted probabilities by app")
        k = stats.kruskal(*[g["Score"].values for _, g in df.groupby("App")])
        save(pd.DataFrame([dict(test="Kruskal-Wallis stars ~ App", H=k.statistic, p=k.pvalue, n=len(df),
                                epsilon_sq=(k.statistic - df['App'].nunique() + 1) / (len(df) - df['App'].nunique()))]),
             "H1_stars_kruskal.csv", title="Stars by app")
        save(dunn(df, "App", "Score"), "H1_stars_dunn.csv", title="Dunn post hoc (Holm)")
        if a.gold_full:
            sub = df[~df["is_junk"]]
            row, _, _ = mnl_block(sub, "H1 mnlogit excluding Junk/Spam", "app")
            _, rr = chi_block(sub, "App", "sent", "H1 chi-square excluding Junk/Spam")
            save(pd.DataFrame([rr, row]), "H1_robust_excluding_junk.csv", title="H1 without the 78 Junk/Spam reviews")
        # within platform
        rows = []
        for pl, sub in df.groupby("Platform"):
            _, r = chi_block(sub, "App", "sent", f"H1 chi-square within {pl}")
            rows.append(r)
        save(pd.DataFrame(rows), "H1_robust_within_platform.csv", title="H1 within platform")
    except Exception as e:
        print("[H1 FAILED]", repr(e))

    # ------------------------------------------------------------------ H2 -------------------------------------- #
    print("\n================ H2: App x Platform ================")
    try:
        rows = []
        for lab, sub in (("H2 App x Platform, full period", df),
                         ("H2 App x Platform, common window", df[df["in_common_window"].astype(bool)])):
            row, fit, X = mnl_block(sub, lab, "interaction")
            rows.append(row)
            if lab.endswith("full period"):
                save(adjusted_probs(fit, X, sub, "Platform", sorted(sub["Platform"].unique())),
                     "H2_adjusted_probabilities_by_platform.csv", title="H2 adjusted probabilities by platform (interaction model)")
        save(pd.DataFrame(rows), "H2_interaction_tests.csv", title="H2 interaction tests")
        rows = []
        for ap_, sub in df.groupby("App"):
            ct, r = chi_block(sub, "Platform", "sent", f"{ap_}: Google Play vs App Store (full period)")
            ct = ct[[s for s in SENT_ORDER if s in ct.columns]]
            r.update({f"pct_{s}_{pl}": float(ct.loc[pl, s] / ct.loc[pl].sum() * 100) for pl in ct.index for s in ct.columns})
            rows.append(r)
            sub2 = sub[sub["in_common_window"].astype(bool)]
            ct2, r2 = chi_block(sub2, "Platform", "sent", f"{ap_}: Google Play vs App Store (common window)")
            ct2 = ct2[[s for s in SENT_ORDER if s in ct2.columns]]
            r2.update({f"pct_{s}_{pl}": float(ct2.loc[pl, s] / ct2.loc[pl].sum() * 100) for pl in ct2.index for s in ct2.columns})
            rows.append(r2)
        t = pd.DataFrame(rows)
        t["p_holm_within_period"] = np.nan
        for per in ("full period", "common window"):
            msk = t["test"].str.contains(per)
            t.loc[msk, "p_holm_within_period"] = holm(t.loc[msk, "p_asymptotic"])
        save(t, "H2_platform_contrasts_by_app.csv", title="H2 platform contrasts within app")
        sp = (df.groupby(["App", "Platform"]).agg(n=("review_id", "size"), mean_stars=("Score", "mean")).reset_index())
        save(sp, "H2_stars_by_app_platform.csv", show=False)
        kk = []
        for ap_, sub in df.groupby("App"):
            g1 = sub.loc[sub["Platform"] == "Google Play", "Score"]; g2 = sub.loc[sub["Platform"] == "App Store", "Score"]
            u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            kk.append(dict(App=ap_, n_gp=len(g1), n_as=len(g2), mean_gp=g1.mean(), mean_as=g2.mean(), U=u.statistic, p=u.pvalue))
        kk = pd.DataFrame(kk); kk["p_holm"] = holm(kk["p"])
        save(kk, "H2_stars_platform_mannwhitney.csv", title="Stars: Google Play vs App Store within app")
        if a.gold_full:
            sub = df[~df["is_junk"]]
            row, _, _ = mnl_block(sub, "H2 interaction excluding Junk/Spam", "interaction")
            save(pd.DataFrame([row]), "H2_robust_excluding_junk.csv", title="H2 without Junk/Spam")
    except Exception as e:
        print("[H2 FAILED]", repr(e))

    # ------------------------------------------------------------------ H3 -------------------------------------- #
    print("\n================ H3: topics of negative reviews by app ================")
    try:
        neg = an[(an["sent"] == "Negative")].copy()
        neg = neg[neg["Topic"] >= 0]
        neg["topic_lab"] = neg["Topic"].astype(int).astype(str)
        specs = [("H3 final topics, full period", neg),
                 ("H3 final topics, common window", neg[neg["in_common_window"].astype(bool)])]
        raw_neg = df[df["in_topic_corpus"] & (df["sent"] == "Negative") & df["dimension_raw"].isin(DIMENSIONS)].copy()
        raw_neg = raw_neg[raw_neg["topic_raw"] >= 0]
        raw_neg["topic_lab"] = raw_neg["topic_raw"].astype(int).astype(str)
        specs.append(("H3 unreduced HDBSCAN topics (outliers dropped)", raw_neg))
        rows = []
        for lab, sub in specs:
            ct, r = chi_block(sub, "topic_lab", "App", lab, perm=a.n_perm, rng=rng)
            r["n_topics"] = ct.shape[0]
            rows.append(r)
            if lab == specs[0][0]:
                ct_main = ct
                z = adj_resid(ct)
                tn = cb.set_index(cb["Topic"].astype(int).astype(str))["topic_name_en"]
                tdim = cb.set_index(cb["Topic"].astype(int).astype(str))["dimension"]
                outc = pd.concat([ct.add_prefix("n_"), (ct.div(ct.sum(0), axis=1) * 100).add_prefix("pct_col_"),
                                  z.add_prefix("z_")], axis=1)
                outc.insert(0, "dimension", tdim.reindex(outc.index).values)
                outc = outc.loc[sorted(outc.index, key=int)]
                outc.insert(0, "topic_name", tn.reindex(outc.index).values)
                save(outc.reset_index().rename(columns={"topic_lab": "Topic"}), "H3_topic_by_app_negative.csv",
                     title="H3 topic x app among Negative reviews (counts, column %, adjusted residuals)")
        if "brand_mention_cluster" in cb.columns:
            bt = set(cb.loc[cb["brand_mention_cluster"].astype(str).str.lower() == "yes", "Topic"].astype(int))
            if bt:
                sub = neg[~neg["Topic"].astype(int).isin(bt)]
                ct_, r_ = chi_block(sub, "topic_lab", "App", f"H3 excluding brand-mention topics {sorted(bt)}", perm=a.n_perm, rng=rng)
                r_["n_topics"] = ct_.shape[0]
                rows.append(r_)
        save(pd.DataFrame(rows), "H3_tests.csv", title="H3 chi-square with permutation p")
        ct, r = chi_block(neg, "dimension", "App", "Dimension x App among Negative reviews", perm=a.n_perm, rng=rng)
        save(pd.concat([ct.add_prefix("n_"), (ct.div(ct.sum(0), axis=1) * 100).add_prefix("pct_col_"),
                        adj_resid(ct).add_prefix("z_")], axis=1).reset_index(), "H3_dimension_by_app_negative.csv",
             title="Dimension x App among Negative reviews")
        save(pd.DataFrame([r]), "H3_dimension_test.csv", title="Dimension x App test (Table 4.14 source)")
    except Exception as e:
        print("[H3 FAILED]", repr(e))

    # ------------------------------------------------------------------ H4a ------------------------------------- #
    print("\n================ H4a: ordered logit, rating ~ dimension ================")
    h4 = None
    try:
        d4, X4, res4, tab4, dims4 = run_h4a(an, "H4a main")
        h4 = (d4, X4, res4, dims4)
        thr = [n for n in res4.params.index if "/" in str(n)]
        save(tab4, "H4a_odds_ratios.csv", title=f"H4a odds ratios vs {REF_DIM} (reference); dimensions with >=30 reviews: {dims4}")
        cfg["h4a"] = dict(n=len(d4), llf=float(res4.llf), dims=dims4, n_clusters=int(d4["cluster"].nunique()))
        # all coefficients (controls too)
        cn = [n for n in res4.params.index if n in X4.columns]
        save(coef_table(res4, cn), "H4a_all_coefficients.csv", show=False)
        # pairwise contrasts between dimensions (Holm)
        names = [f"dim_{k}" for k in dims4 if k != REF_DIM]
        pr = []
        for u, v in combinations(dims4, 2):
            w = {k: 0.0 for k in names}
            if u != REF_DIM: w[f"dim_{u}"] += 1
            if v != REF_DIM: w[f"dim_{v}"] -= 1
            est, se, p = wald_linear(res4, names, [w[k] for k in names])
            pr.append(dict(dim_a=u, dim_b=v, log_OR_a_minus_b=est, se=se, OR_a_vs_b=np.exp(est), p=p))
        pr = pd.DataFrame(pr); pr["p_holm"] = holm(pr["p"])
        save(pr, "H4a_pairwise_contrasts.csv", title="H4a pairwise contrasts (Holm)")
        # adjusted predicted probabilities (controls at their means)
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        Xm = X4.mean()
        rows = []
        for k in dims4:
            x = Xm.copy()
            for c in names: x[c] = 0.0
            if k != REF_DIM: x[f"dim_{k}"] = 1.0
            pp = res4.model.predict(res4.params.values, exog=x.values.reshape(1, -1))[0]
            rows.append(dict(dimension=k, **{f"P(rating={i + 1})": pp[i] for i in range(5)},
                             expected_rating=float(np.dot(pp, np.arange(1, 6)))))
        save(pd.DataFrame(rows), "H4a_predicted_probabilities.csv", title="H4a adjusted predicted probabilities")
        # proportional-odds check
        import statsmodels.api as sm
        rows = []
        for cut in (1, 2, 3, 4):
            yb = (d4["Score"].astype(int) > cut).astype(float).values
            try:
                rb = sm.Logit(yb, sm.add_constant(X4, has_constant="add")).fit(disp=0, maxiter=300, cov_type="cluster",
                                                                              cov_kwds={"groups": d4["cluster"].values})
                for nme in names:
                    rows.append(dict(cut=f"rating > {cut}", term=nme, coef=rb.params[nme], se=rb.bse[nme], p=rb.pvalues[nme]))
            except Exception as e:
                print("  [cut", cut, "failed]", e)
        save(pd.DataFrame(rows), "H4a_proportional_odds_check.csv", title="H4a coefficients of separate binary logits at each cut-point")
        hl = d4[d4["Score"].isin([1, 2, 4, 5])]
        Xh = drop_constant(X4.loc[hl.index])
        rb = sm.Logit((hl["Score"] >= 4).astype(float).values, sm.add_constant(Xh, has_constant="add")).fit(
            disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": hl["cluster"].values})
        tb = coef_table(rb, [n for n in names if n in rb.params.index]); tb.insert(0, "model", "binary logit high(4-5) vs low(1-2)")
        save(tb, "H4a_binary_high_vs_low.csv", title="H4a robustness: high vs low rating")
        # VIF
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xv = sm.add_constant(X4.loc[:, [c for c in X4.columns if not c.startswith("Q_")]], has_constant="add")
        vif = pd.DataFrame({"term": Xv.columns, "VIF": [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]})
        save(vif[vif.term != "const"], "H4a_vif.csv", title="H4a variance inflation factors (quarter dummies omitted)")
        # sensitivity: move ONE topic to its alt_dimension
        if not a.skip_sensitivity:
            alt = cb[(cb["alt_dimension"].isin(DIMENSIONS)) & (cb["Topic"] >= 0) & (cb["dimension"].isin(DIMENSIONS))]
            rows = []
            for r in alt.itertuples():
                s = an.copy()
                s.loc[s["Topic"] == r.Topic, "dimension"] = r.alt_dimension
                try:
                    _, _, _, t_, _ = run_h4a(s, f"T{r.Topic}: {r.dimension} -> {r.alt_dimension}")
                    t_["moved_topic"] = r.Topic; t_["from"] = r.dimension; t_["to"] = r.alt_dimension
                    rows.append(t_)
                except Exception as e:
                    print(f"  [sensitivity T{r.Topic} failed] {e}")
            if rows:
                sens = pd.concat(rows)
                save(sens[["model", "moved_topic", "from", "to", "term", "OR", "OR_lo", "OR_hi", "p", "n"]],
                     "H4a_sensitivity_topic_reassignment.csv", show=False)
                print("\n--- H4a sensitivity: odds ratios for System Availability and Fulfillment when ONE topic is moved")
                show = sens[sens["term"].isin(["dim_System Availability", "dim_Fulfillment", "dim_Efficiency"])]
                print(show.pivot_table(index="model", columns="term", values="OR").round(3).to_string())
        # raw HDBSCAN assignment
        rawan = df[df["in_topic_corpus"] & df["dimension_raw"].isin(DIMENSIONS)].copy()
        rawan["dimension"] = rawan["dimension_raw"]
        _, _, _, t_, _ = run_h4a(rawan, "H4a on unreduced HDBSCAN topics")
        save(t_, "H4a_robust_unreduced_hdbscan.csv", title="H4a on the unreduced HDBSCAN assignment")
        cw = an[an["in_common_window"].astype(bool)]
        _, _, _, t_, _ = run_h4a(cw, "H4a common window")
        save(t_, "H4a_robust_common_window.csv", title="H4a in the common window")
        if "brand_mention_cluster" in cb.columns:
            bt = set(cb.loc[cb["brand_mention_cluster"].astype(str).str.lower() == "yes", "Topic"].astype(int))
            if bt:
                _, _, _, t_, _ = run_h4a(an[~an["Topic"].astype(int).isin(bt)], f"H4a excluding brand-mention topics {sorted(bt)}")
                save(t_, "H4a_robust_excluding_brand_topics.csv", title="H4a without the brand-mention topics")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[H4a FAILED]", repr(e))

    # ------------------------------------------------------------------ H4b ------------------------------------- #
    print("\n================ H4b: asymmetry of negative vs positive reviews ================")
    try:
        d4b = an.copy()
        cell = d4b.groupby(["dimension", "sent"]).size().unstack(fill_value=0)
        print(cell.to_string())
        dims_b = [k for k in DIMENSIONS if k in cell.index and cell.loc[k].sum() >= 30]
        keep_neg = [k for k in dims_b if cell.loc[k].get("Negative", 0) >= 10]
        keep_pos = [k for k in dims_b if cell.loc[k].get("Positive", 0) >= 10]
        cols = {}
        for k in keep_neg: cols[f"neg_{k}"] = ((d4b["dimension"] == k) & (d4b["sent"] == "Negative")).astype(float)
        for k in keep_pos: cols[f"pos_{k}"] = ((d4b["dimension"] == k) & (d4b["sent"] == "Positive")).astype(float)
        P = pd.DataFrame(cols, index=d4b.index)
        # reviews in a (dimension, polarity) cell below 10 are removed so that they do not contaminate the reference group
        small = pd.Series(False, index=d4b.index)
        for k in dims_b:
            for s in ("Negative", "Positive"):
                if cell.loc[k].get(s, 0) < 10:
                    small |= (d4b["dimension"] == k) & (d4b["sent"] == s)
        small |= ~d4b["dimension"].isin(dims_b)
        d4b, P = d4b[~small], P[~small]
        X = drop_constant(pd.concat([P, controls_matrix(d4b)], axis=1))
        resb = fit_ordered(pd.Series(d4b["Score"].astype(int).values, index=X.index), X, d4b["cluster"].values)
        rows = []
        for k in dims_b:
            row = dict(dimension=k, n_neg=int(cell.loc[k].get("Negative", 0)), n_pos=int(cell.loc[k].get("Positive", 0)))
            if f"neg_{k}" in resb.params.index: row.update(b_neg=resb.params[f"neg_{k}"], se_neg=resb.bse[f"neg_{k}"], p_neg=resb.pvalues[f"neg_{k}"])
            if f"pos_{k}" in resb.params.index: row.update(b_pos=resb.params[f"pos_{k}"], se_pos=resb.bse[f"pos_{k}"], p_pos=resb.pvalues[f"pos_{k}"])
            if f"neg_{k}" in resb.params.index and f"pos_{k}" in resb.params.index:
                est, se, p = wald_linear(resb, [f"neg_{k}", f"pos_{k}"], [1, 1])
                row.update(sum_b=est, se_sum=se, p_wald_sum_zero=p, asymmetry_index=abs(row["b_neg"]) / abs(row["b_pos"]))
            rows.append(row)
        tb = pd.DataFrame(rows)
        if "p_wald_sum_zero" in tb:
            ok = tb["p_wald_sum_zero"].notna()
            tb.loc[ok, "p_wald_holm"] = holm(tb.loc[ok, "p_wald_sum_zero"])
        tb["n_model"] = len(d4b)
        save(tb, "H4b_asymmetry.csv", title="H4b asymmetry (reference group: Neutral reviews)")
        cfg["h4b"] = dict(n=len(d4b), n_reference_neutral=int((d4b['sent'] == 'Neutral').sum()))
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[H4b FAILED]", repr(e))

    # ------------------------------------------------------------------ H5 -------------------------------------- #
    print("\n================ H5: net sentiment by dimension x app (bootstrap) ================")
    try:
        sc = an["sent"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
        an = an.assign(sv=sc)
        apps = sorted(an["App"].unique())
        rows, comps = [], []
        for k in [x for x in DIMENSIONS if x in set(an["dimension"])]:
            cells = {}
            for ap_ in apps:
                s = an[(an["dimension"] == k) & (an["App"] == ap_)]["sv"]
                n = len(s)
                if n:
                    cells[ap_] = (n, (s == 1).sum(), (s == 0).sum(), (s == -1).sum())
                    net = ((s == 1).mean() - (s == -1).mean()) * 100
                    pr_ = np.array(cells[ap_][1:], float) / n
                    boot = rng.multinomial(n, pr_, size=a.n_boot)
                    bn = (boot[:, 0] - boot[:, 2]) / n * 100
                    rows.append(dict(dimension=k, App=ap_, n=n, pct_pos=(s == 1).mean() * 100, pct_neu=(s == 0).mean() * 100,
                                     pct_neg=(s == -1).mean() * 100, net_sentiment=net,
                                     ci_lo=np.percentile(bn, 2.5), ci_hi=np.percentile(bn, 97.5),
                                     estimable=n >= a.min_cell))
            for u, v in combinations(apps, 2):
                if u in cells and v in cells and cells[u][0] >= a.min_cell and cells[v][0] >= a.min_cell:
                    bu = rng.multinomial(cells[u][0], np.array(cells[u][1:], float) / cells[u][0], size=a.n_boot)
                    bv = rng.multinomial(cells[v][0], np.array(cells[v][1:], float) / cells[v][0], size=a.n_boot)
                    du = (bu[:, 0] - bu[:, 2]) / cells[u][0] * 100
                    dv = (bv[:, 0] - bv[:, 2]) / cells[v][0] * 100
                    diff = du - dv
                    obs = ((cells[u][1] - cells[u][3]) / cells[u][0] - (cells[v][1] - cells[v][3]) / cells[v][0]) * 100
                    p = min(1.0, 2 * min((np.sum(diff <= 0) + 1) / (a.n_boot + 1), (np.sum(diff >= 0) + 1) / (a.n_boot + 1)))
                    comps.append(dict(dimension=k, app_a=u, app_b=v, n_a=cells[u][0], n_b=cells[v][0], diff_net_a_minus_b=obs,
                                      ci_lo=np.percentile(diff, 2.5), ci_hi=np.percentile(diff, 97.5), p_boot=p))
        save(pd.DataFrame(rows), "H5_net_sentiment.csv", title="H5 net sentiment by dimension x app (95% bootstrap CI)")
        cp = pd.DataFrame(comps)
        if len(cp):
            cp["p_holm"] = holm(cp["p_boot"]); cp["p_bh"] = bh(cp["p_boot"])
            cp["ci_excludes_0"] = (cp["ci_lo"] > 0) | (cp["ci_hi"] < 0)
        save(cp, "H5_pairwise_differences.csv", title="H5 pairwise differences (Holm across all comparisons)")
        tot = an.groupby("dimension")["sv"].agg(n="size", net_sentiment=lambda s: ((s == 1).mean() - (s == -1).mean()) * 100).reset_index()
        save(tot, "H5_net_sentiment_by_dimension_overall.csv", show=False)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[H5 FAILED]", repr(e))

    # ------------------------------------------------------------------ finish ---------------------------------- #
    import importlib.metadata as md
    cfg["versions"] = {p: md.version(p) for p in ("statsmodels", "scipy", "pandas", "numpy")}
    (OUT / "analysis_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    print("\nDone. Tables in", OUT.resolve())
    try:
        import shutil
        shutil.make_archive(str(OUT.parent / "outputs_stats"), "zip", OUT)
        print("Zipped:", (OUT.parent / "outputs_stats.zip").resolve())
    except Exception as e:
        print("[zip skipped]", e)


if __name__ == "__main__":
    main()
