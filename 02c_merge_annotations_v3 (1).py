"""
PHASE 2c (v3) - Merge annotations, compute Cohen's kappa, adjudicate, build the gold-labelled file.

v3: accepts .csv or .xlsx/.xls, exact file paths (--file1/--file2), and needs gold_key.csv only in --mode full.

Colab example (pilot):
    !python 02c_merge_annotations_v3.py --mode pilot --file1 gold_pilot_coder1.csv --file2 gold_pilot_coder2.csv
Full (Coder 1 = pilot + main; Coder 2 = pilot + extra; several files are concatenated):
    !python 02c_merge_annotations_v3.py --mode full --file1 gold_pilot_coder1.csv gold_main_coder1.csv \\
        --file2 gold_pilot_coder2.csv gold_main_coder2_extra.csv --key gold_key.csv

FROZEN ANNOTATION GUIDELINE (frozen after the pilot round of 100 reviews; applies to ALL remaining annotation)
  Labels: Positive | Negative | Neutral | Mixed | Junk/Spam. `lean_if_mixed` (Positive/Negative/Neutral) is required
  when label = Mixed and left empty otherwise.
  Base definitions
    Positive  : satisfaction, praise, good experience.
    Negative  : complaints, bugs, frustration, poor experience.
    Neutral   : objective questions / statements of fact without clear sentiment.
    Mixed     : BOTH clear positive and clear negative sentiment in one review.
    Junk/Spam : referral codes, fake-loan links, completely irrelevant text.
  Rules added after adjudicating the pilot disagreements
    R1 Gibberish / nonsense : meaningless character sequences (e.g. "cc", "hdyux") -> Junk/Spam.
    R2 Spoken-Vietnamese "lỗi không" : at the end of a sentence it means "full of errors / a lot of errors"
       (NOT "no errors") -> Negative.
    R3 Constructive feature request : the app is judged good but a specific feature is missing, phrased
       "if it had [X] it would be great/ok" (e.g. "Thêm cập nhật CCCD trực tiếp trên ứng dụng nữa là quá ok")
       -> Mixed, lean_if_mixed = Positive.
    R4 Limitation / inconvenience : a strict limitation or inconvenience, typically with "chỉ đc" (only allowed to)
       (e.g. "2ae sinh đôi chỉ đc đk 1 tài khoản") -> Negative.
  Operating conventions used by Coder 2 when applying the rules (not separate rules): a feature suggestion with NO
  expressed appreciation of the app stays Neutral; bare "ok"/"good" is Positive; a question describing a failure
  (e.g. "sao quét NFC k đc vậy") is Negative; a pure question or statement of need is Neutral.

Works with the sheets written by 02a_make_goldset_v3.py:
    gold_id | Platform | text_raw | label | lean_if_mixed | notes

Labels  : Positive, Negative, Neutral, Mixed, Junk/Spam
Rule    : label = Mixed  -> lean_if_mixed (Positive/Negative/Neutral) is REQUIRED
          label != Mixed -> lean_if_mixed must be empty (it is ignored/cleared, with a warning)

KAPPA WORKFLOW
  1. PILOT (100 reviews, both coders):   python 02c_merge_annotations_v3.py --mode pilot
        -> kappa on the 5-class `label` (PRIMARY), + 3-class collapsed, + Junk-vs-rest,
           + agreement on lean_if_mixed among rows that BOTH coders called Mixed (reported separately).
        Decide whether the guideline needs changes; if kappa < ~0.60, revise and repeat.
  2. FULL (400 double-coded):             python 02c_merge_annotations_v3.py --mode full
        -> same statistics on all double-coded rows; disagreements.csv to adjudicate.
  3. After filling final_label / final_lean in adjudication.csv, run --mode full again to build the gold files.

HOW lean_if_mixed IS TREATED IN KAPPA
  * The 5-class kappa uses ONLY `label` (Mixed is one class). lean is NOT part of it.
  * If both coders chose Mixed, their leans are compared separately (agreement %, kappa on 3 classes).
    A lean mismatch does not lower the 5-class kappa, but the row goes to adjudication
    because the 3-class fallback label depends on it.
  * 3-class fallback: label3 = label if label in {Positive,Negative,Neutral}; = lean_if_mixed if Mixed
    (missing lean -> Neutral, flagged); Junk/Spam rows are excluded. Kappa is also reported on label3.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

L5 = ["Positive", "Negative", "Neutral", "Mixed", "Junk/Spam"]
L3 = ["Positive", "Negative", "Neutral"]
NORM = {x.lower(): x for x in L5}
NORM.update({"junk": "Junk/Spam", "spam": "Junk/Spam", "junk/spam": "Junk/Spam"})
ORD = {"Negative": 0, "Neutral": 1, "Positive": 2}     # for linear-weighted kappa


def clean_cell(x, allowed):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    v = NORM.get(str(x).strip().lower(), None)
    if v is None or v not in allowed:
        sys.exit(f"ERROR: invalid value {x!r}; allowed: {allowed}")
    return v


def read_sheet(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path.resolve()}  (cwd = {Path.cwd()})")
    ext = path.suffix.lower()
    if ext == ".csv":
        try:
            d = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            d = pd.read_csv(path, dtype=str, encoding="latin-1")
    elif ext in (".xlsx", ".xlsm", ".xls"):
        xl = pd.ExcelFile(path)
        sheet = "annotate" if "annotate" in xl.sheet_names else xl.sheet_names[0]    # tolerate 'Sheet1'
        d = pd.read_excel(path, sheet_name=sheet, dtype=str)
    else:
        sys.exit(f"ERROR: unsupported file type {ext!r} ({path.name}); use .csv or .xlsx")
    d.columns = [str(c).strip() for c in d.columns]
    need = {"gold_id", "label"}
    if not need <= set(d.columns):
        sys.exit(f"ERROR: {path.name} lacks columns {need - set(d.columns)}")
    if "lean_if_mixed" not in d.columns:
        d["lean_if_mixed"] = ""
    d["gold_id"] = d["gold_id"].astype(str).str.strip()
    d = d[["gold_id", "label", "lean_if_mixed"] + [c for c in ("text_raw", "notes") if c in d.columns]].copy()
    d["label"] = d["label"].map(lambda v: clean_cell(v, L5))
    d["lean_if_mixed"] = d["lean_if_mixed"].map(lambda v: clean_cell(v, L3))
    return d


def tidy(d: pd.DataFrame, who: str) -> pd.DataFrame:
    """Enforce the lean rule and report problems."""
    d = d.copy()
    miss = d[d.label == ""]
    if len(miss):
        sys.exit(f"ERROR [{who}]: {len(miss)} rows have no label, e.g. {list(miss.gold_id[:5])}")
    bad_mixed = d[(d.label == "Mixed") & (d.lean_if_mixed == "")]
    if len(bad_mixed):
        print(f"WARNING [{who}]: {len(bad_mixed)} Mixed rows without lean_if_mixed -> go back and fill them: "
              f"{list(bad_mixed.gold_id[:10])}")
    stray = (d.label != "Mixed") & (d.lean_if_mixed != "")
    if stray.any():
        print(f"WARNING [{who}]: {stray.sum()} non-Mixed rows had a lean value; it was cleared.")
        d.loc[stray, "lean_if_mixed"] = ""
    if d.gold_id.duplicated().any():
        sys.exit(f"ERROR [{who}]: duplicated gold_id")
    return d


def collapse(label, lean):
    """5 labels (+lean) -> 3 classes; Junk -> None. Mixed without lean -> Neutral."""
    if label == "Junk/Spam":
        return None
    if label == "Mixed":
        return lean if lean in L3 else "Neutral"
    return label


def load(files, who):
    return tidy(pd.concat([read_sheet(f) for f in files], ignore_index=True), who)


def kappas(m: pd.DataFrame, title: str, out: Path):
    print(f"\n===== {title}: n = {len(m)} double-coded reviews =====")
    r = {}
    r["agree5"] = (m.label_c1 == m.label_c2).mean()
    r["kappa5"] = cohen_kappa_score(m.label_c1, m.label_c2)
    print(f"[PRIMARY] 5-class `label`: agreement {r['agree5']:.1%}  Cohen's kappa = {r['kappa5']:.3f}")
    cm = pd.crosstab(m.label_c1, m.label_c2).reindex(index=L5, columns=L5, fill_value=0)
    print("Confusion matrix (rows = coder 1, cols = coder 2):\n", cm.to_string())
    cm.to_csv(out / "confusion_5class.csv")
    # Junk vs rest
    j1, j2 = m.label_c1 == "Junk/Spam", m.label_c2 == "Junk/Spam"
    r["kappa_junk"] = cohen_kappa_score(j1, j2) if (j1.any() or j2.any()) else np.nan
    print(f"Junk/Spam vs rest kappa = {r['kappa_junk']:.3f}")
    # lean agreement where both said Mixed
    both = m[(m.label_c1 == "Mixed") & (m.label_c2 == "Mixed")]
    r["n_both_mixed"] = len(both)
    if len(both):
        r["lean_agree"] = (both.lean_c1 == both.lean_c2).mean()
        r["lean_kappa"] = cohen_kappa_score(both.lean_c1, both.lean_c2) if both.lean_c1.nunique() > 1 or both.lean_c2.nunique() > 1 else np.nan
        print(f"Both coders chose Mixed in {len(both)} rows; lean_if_mixed agreement {r['lean_agree']:.1%}"
              f" (kappa {r['lean_kappa']:.3f}; unreliable when n is small)")
    else:
        print("No review was labelled Mixed by both coders -> lean agreement not computable.")
    # 3-class fallback
    m3 = m.assign(l1=[collapse(a, b) for a, b in zip(m.label_c1, m.lean_c1)],
                  l2=[collapse(a, b) for a, b in zip(m.label_c2, m.lean_c2)])
    m3 = m3[m3.l1.notna() & m3.l2.notna()]
    r["n3"] = len(m3)
    r["agree3"] = (m3.l1 == m3.l2).mean()
    r["kappa3"] = cohen_kappa_score(m3.l1, m3.l2)
    r["kappa3_w"] = cohen_kappa_score(m3.l1.map(ORD), m3.l2.map(ORD), weights="linear")
    print(f"[FALLBACK] 3-class (Mixed -> lean; Junk excluded, n = {len(m3)}): agreement {r['agree3']:.1%}, "
          f"kappa = {r['kappa3']:.3f}, linear-weighted kappa = {r['kappa3_w']:.3f}")
    pd.Series(r).to_csv(out / "kappa_summary.csv", header=["value"])
    return r


def main():
    ap = argparse.ArgumentParser(description="Kappa + merge of two coders' annotation files (.csv or .xlsx).")
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--file1", nargs="+", required=True, help="Coder 1 file(s); in full mode: pilot + main")
    ap.add_argument("--file2", nargs="+", required=True, help="Coder 2 file(s); in full mode: pilot + extra")
    ap.add_argument("--key", default="gold_key.csv", help="gold_key.csv (needed only in --mode full)")
    ap.add_argument("--outdir", default="merged")
    ap.add_argument("--adjudication", default=None, help="adjudication.csv (default: <outdir>/adjudication.csv)")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    c1 = load(a.file1, "coder1")
    c2 = load(a.file2, "coder2")
    key = None
    if a.mode == "full":
        if not Path(a.key).exists():
            sys.exit(f"ERROR: --mode full needs gold_key.csv; not found at {Path(a.key).resolve()}")
        key = pd.read_csv(a.key)
        if len(c1) != len(key):
            sys.exit(f"ERROR: coder 1 has {len(c1)} rows, gold_key has {len(key)}")

    m = c1.merge(c2, on="gold_id", suffixes=("_c1", "_c2"))
    m = m.rename(columns={"lean_if_mixed_c1": "lean_c1", "lean_if_mixed_c2": "lean_c2"})
    if len(m) == 0:
        sys.exit("ERROR: no gold_id shared by the two coders")
    kappas(m, "PILOT" if a.mode == "pilot" else "FULL OVERLAP", out)

    # ---- disagreements: label differs, OR both Mixed but lean differs ----
    lab_dis = m.label_c1 != m.label_c2
    lean_dis = (m.label_c1 == "Mixed") & (m.label_c2 == "Mixed") & (m.lean_c1 != m.lean_c2)
    dis = m[lab_dis | lean_dis].copy()
    dis["disagreement_type"] = np.where(lab_dis[dis.index], "label", "lean_only")
    if "text_raw_c1" in dis:
        dis = dis.rename(columns={"text_raw_c1": "text_raw"})
    dis["final_label"], dis["final_lean"] = "", ""
    cols = [c for c in ["gold_id", "text_raw", "disagreement_type", "label_c1", "lean_c1", "label_c2", "lean_c2",
                        "final_label", "final_lean"] if c in dis.columns]
    fn = "pilot_disagreements.csv" if a.mode == "pilot" else "disagreements.csv"
    dis[cols].to_csv(out / fn, index=False, encoding="utf-8-sig")
    print(f"\n{len(dis)} rows to discuss/adjudicate -> {out / fn}")
    if a.mode == "pilot":
        print("Pilot only: revise the guideline if needed, then label the rest. Nothing else written.")
        return

    # ---- adjudication ----
    adj_path = Path(a.adjudication) if a.adjudication else out / "adjudication.csv"
    if not adj_path.exists():
        dis[cols].to_csv(adj_path, index=False, encoding="utf-8-sig")
        print(f"Template written: {adj_path}\n-> fill `final_label` (and `final_lean` if final_label = Mixed), then rerun.")
        return
    adj = pd.read_csv(adj_path, dtype=str).fillna("")
    adj["final_label"] = adj.final_label.map(lambda v: clean_cell(v, L5))
    adj["final_lean"] = adj.final_lean.map(lambda v: clean_cell(v, L3))
    open_rows = adj[adj.final_label == ""]
    if len(open_rows):
        sys.exit(f"ERROR: {len(open_rows)} adjudication rows have no final_label, e.g. {list(open_rows.gold_id[:5])}")
    if ((adj.final_label == "Mixed") & (adj.final_lean == "")).any():
        sys.exit("ERROR: adjudicated Mixed rows need final_lean")
    adj.loc[adj.final_label != "Mixed", "final_lean"] = ""
    need = set(m.loc[lab_dis | lean_dis, "gold_id"])
    if need - set(adj.gold_id):
        sys.exit(f"ERROR: adjudication.csv lacks {len(need - set(adj.gold_id))} disagreeing gold_id(s)")
    amap = adj.set_index("gold_id")[["final_label", "final_lean"]]

    # ---- final labels: coder 1 by default, agreed value for double-coded, adjudicated if they differed ----
    g = key.merge(c1[["gold_id", "label", "lean_if_mixed"]], on="gold_id", how="left")
    g = g.rename(columns={"label": "label5", "lean_if_mixed": "lean"})
    g["adjudicated"] = g.gold_id.isin(amap.index)
    g.loc[g.adjudicated, "label5"] = g.loc[g.adjudicated, "gold_id"].map(amap.final_label).values
    g.loc[g.adjudicated, "lean"] = g.loc[g.adjudicated, "gold_id"].map(amap.final_lean).values
    g["label3"] = [collapse(a_, b_) for a_, b_ in zip(g.label5, g.lean)]
    g["mixed_no_lean"] = (g.label5 == "Mixed") & (g.lean == "")
    g.to_csv(out / "gold_labeled_full.csv", index=False, encoding="utf-8-sig")
    g3 = g[g.label3.notna()].drop(columns=["label3"]).assign(label=g.label3[g.label3.notna()])
    g3.to_csv(out / "gold_labeled.csv", index=False, encoding="utf-8-sig")

    print(f"\ngold_labeled_full.csv : {len(g)} rows (5-class `label5`, `lean`, 3-class `label3`)")
    print(f"gold_labeled.csv      : {len(g3)} rows (3-class `label`, Junk/Spam removed)")
    if g.mixed_no_lean.any():
        print(f"WARNING: {g.mixed_no_lean.sum()} Mixed rows without lean were mapped to Neutral.")
    if "weight" in g:
        for col, nm in (("label5", "5-class"), ("label3", "3-class")):
            s = g.dropna(subset=[col]).groupby(col).weight.sum()
            print(f"Population-weighted share, {nm}:\n", (s / s.sum()).round(4).to_string())


if __name__ == "__main__":
    main()
