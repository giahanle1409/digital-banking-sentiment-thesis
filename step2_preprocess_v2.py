#!/usr/bin/env python3
"""
PHASE 1 (v2, unified) - Data preprocessing for the Timo / Vikki / Cake app-review corpus (Google Play + App Store)
==========================================================================
Input : dataset_reviews_raw_merged.csv  (App, Platform, Date, Score, Content, Thumbs_Up)  <- output of step1_merge.py
Output: data/reviews_clean.csv|.parquet  - master table (one row per review) with
            text_raw      original text (never modified; for audit / annotation display)
            text_clean    normalised text (NFC, teencode, elongation, PII masked)
            text_seg      word-segmented text with '_' joins  -> PhoBERT input
            text_embed    clean text without placeholders/emoji tokens -> sentence embeddings
            text_topic    segmented text without punctuation/placeholders -> c-TF-IDF vocabulary
        data/teencode_audit.csv          how often each abbreviation was replaced (verify by hand)
        data/candidate_abbreviations.csv frequent short tokens NOT in the dictionary (extend it)
        data/dropped_rows.csv            rows removed and why
        data/spam_removed.csv            promotional spam removed (TinChac ads, fake loan links) - audit trail
        data/spam_suspect_manual_review.csv  weaker spam signals, kept but listed for you to check
        data/preprocessing_report.json   numbers to quote in Chapter 3 / 4.1 (sample-flow table)

Usage:
    python step2_preprocess_v2.py --input dataset_reviews_raw_merged.csv --outdir data
    python step2_preprocess_v2.py --self-test                # unit checks on the cleaning rules only

Design decisions (all documented so you can defend them in Chapter 3):
  * Short generic reviews ("tốt", "ok") are KEPT for sentiment, only flagged out of the topic corpus.
  * Only true scrape duplicates (same App+Platform+Timestamp+Content) are dropped. Identical short
    texts written by different users are legitimate observations and are kept.
  * Ambiguous abbreviations (nh, tt, bh, kh, dt ...) are NOT auto-replaced; they are listed in
    candidate_abbreviations.csv for manual decision.
  * Promotional spam (templated ads, obfuscated loan-site links) is removed: it is almost all 5-star and off-topic.
  * text_raw is preserved; every transformation is reproducible from it.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# 0. CONFIG
# --------------------------------------------------------------------------- #
REQUIRED_COLS = ["App", "Platform", "Date", "Score", "Content", "Thumbs_Up"]
COMMON_WINDOW_START = "2025-03-01"   # Vikki: launch burst in Jul-2024, then <80 reviews/month until Mar-2025; DongA renamed Vikki in Feb-2025
TOPIC_MIN_WORDS = 4                  # reviews with fewer words are excluded from the topic corpus only
LOCAL_TZ = "Asia/Ho_Chi_Minh"

# Teencode / abbreviation dictionary (derived from token frequencies in THIS corpus; extend after
# inspecting candidate_abbreviations.csv). Keys must be lowercase, single tokens.
TEENCODE = {
    # negation
    "ko": "không", "k": "không", "kg": "không", "khg": "không", "hok": "không", "hông": "không",
    "hem": "không", "hk": "không", "kô": "không", "kh0ng": "không",
    # "được"
    "đc": "được", "dc": "được",   # NB: "đk" is ambiguous (điều kiện / đăng ký / được) -> left for manual review
    # banking / app vocabulary
    "cskh": "chăm sóc khách hàng", "ngh": "ngân hàng", "tk": "tài khoản", "stk": "số tài khoản",
    "sdt": "số điện thoại", "sđt": "số điện thoại", "đt": "điện thoại",
    "mk": "mật khẩu", "gd": "giao dịch", "ck": "chuyển khoản", "nv": "nhân viên",
    "dv": "dịch vụ", "tg": "thời gian",
    # people / pronouns / fillers
    "mn": "mọi người", "ae": "anh em", "ng": "người", "mik": "mình", "mih": "mình",
    "e": "em",
    # grammar words
    "j": "gì", "vs": "với", "z": "vậy", "v": "vậy", "r": "rồi", "ròi": "rồi",
    "bt": "bình thường", "trc": "trước", "wa": "quá", "qá": "quá", "cx": "cũng",
    "cg": "cũng", "nhg": "nhưng", "nhưg": "nhưng", "đg": "đang", "đag": "đang",
    "chx": "chưa", "lun": "luôn", "lm": "làm", "bik": "biết", "ns": "nói",
    "thik": "thích", "ntn": "như thế nào", "ib": "nhắn tin", "nt": "nhắn tin",
    "cmt": "bình luận", "sài": "xài",
    # thanks / ok variants -> canonical
    "tks": "cảm ơn", "thks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
    "oke": "ok", "okie": "ok", "okay": "ok", "oki": "ok", "okê": "ok",
}
# Tokens that must never be touched even if they look like abbreviations
PROTECTED = {"otp", "nfc", "qr", "cccd", "vneid", "app", "sms", "atm", "vat", "ok"}

# Emoji -> polarity placeholder (≈3% of reviews contain emoji). Unknown emoji are dropped.
EMOJI_POS = set("👍❤♥🥰🌟💯⭐🙏😀😍😘🌹👌🤩😄😆😁😊🩷💖🥹😌💕👏😃😇🤗✨💪🎉🌸💝✯✮☆😻")
EMOJI_NEG = set("😡🤬👎😭😔😢😪🙄😏💩🤮😤😠😞😩😱💀🖕😒😖😫👿💔🤡")
EMOJI_NEU = set("🙂🙃🤔🤣😂😅😮🥲😬😑😐🤨🥴")  # ambiguous / often sarcastic
EMOJI_ANY = re.compile(r"[\U0001F000-\U0001FAFF☀-➿⭐⬆❤♡☆✯✮]")
EMOJI_JUNK = re.compile(r"[️‍\U0001F3FB-\U0001F3FF\U0001F1E6-\U0001F1FF]")  # selectors, ZWJ, skin tones, flags

# Old-style -> new-style tone-mark placement (khoá -> khóa, thuỷ -> thủy). 'qu' is excluded: quý stays quý.
TONE_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
# Only open syllables are re-marked (hoà -> hòa). With a final consonant the tone already sits on the
# second vowel in BOTH orthographies (khoản, toán, hoàn, huỳnh, thoại) and must not be moved.
_TONE_RE = re.compile(r"(?<![q])(" + "|".join(TONE_MAP) + r")(?![^\W\d_])")

VN_DIACRITICS = re.compile(
    r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]")

# --------------------------------------------------------------------------- #
# 1. TEXT CLEANING
# --------------------------------------------------------------------------- #
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)[\d .-]{8,12}\d(?!\d)")
_LONGNUM = re.compile(r"(?<!\d)\d{9,}(?!\d)")                 # CCCD, account numbers
_CENSOR = re.compile(r"(?<!\S)\*{2,}(?!\S)")                  # store-masked profanity: ***
_ELONG3 = re.compile(r"([^\W\d_])\1{2,}")                      # >=3 identical letters
# elongation after a toned vowel repeats the BARE vowel: 'quáaaa', 'ngonnn', 'ơiii' -> handle 'á'+'aaa'
_TONED_ELONG = re.compile(
    "(?:([àáảãạăằắẳẵặâầấẩẫậ])a+|([èéẻẽẹêềếểễệ])e+|([ìíỉĩị])i+|([òóỏõọôồốổỗộơờớởỡợ])o+|([ùúủũụưừứửữự])u+|([ỳýỷỹỵ])y+)")
_DOUBLE = re.compile(r"([^\W\d_])\1+")
_PUNCT_RUN = re.compile(r"([!?,;:])\1+")
_DOTS = re.compile(r"\.{2,}")
_MONEY_K = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s?k(?!\w)")   # 50k -> 50 nghìn  (BEFORE 'k' -> 'không')
_MONEY_TR = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s?(?:tr|triệu)(?!\w)")
_SPACE_PUNCT = re.compile(r"(?<!\d)([.,!?;:()%])|([.,!?;:()%])(?!\d)")
_WORD = re.compile(r"\w+", re.UNICODE)
_SAIGON = re.compile(r"\bsài(?=\s+gòn)")
_CTRL = re.compile(r"[​-‏⁠﻿­]")


def _emoji_to_token(m: re.Match) -> str:
    ch = m.group()
    if ch in EMOJI_POS:
        return " emojipos "
    if ch in EMOJI_NEG:
        return " emojineg "
    if ch in EMOJI_NEU:
        return " emojineu "
    return " "


def normalize_text(s: object, stats: Counter | None = None) -> str:
    """Raw review -> clean text. `stats` (optional) collects counts for the audit tables."""
    stats = stats if stats is not None else Counter()
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)                       # (1) Unicode NFC
    s = _CTRL.sub("", s).replace("\xa0", " ")
    s = s.lower()
    # PII / noise masking (public data, but we do not store identifiers: Chapter 3.2.5)
    s = _URL.sub(" url ", s)
    s = _EMAIL.sub(" email ", s)
    s = _PHONE.sub(" phone ", s)
    s = _LONGNUM.sub(" number ", s)
    # emoji -> polarity tokens
    s = EMOJI_JUNK.sub("", s)
    if EMOJI_ANY.search(s):
        stats["has_emoji"] += 1
        s = EMOJI_ANY.sub(_emoji_to_token, s)
    # markdown / censor stars. '***' alone = masked profanity; remaining '*' are markdown bold/bullets
    s = _CENSOR.sub(" từ tục ", s)
    s = s.replace("*", " ")
    # '_' is the word-joiner in PhoBERT input; user underscores would corrupt segmentation
    s = s.replace("_", " ")
    s = re.sub(r"[~^|<>#@=+\[\]{}\\`\"“”‘’'«»/]", " ", s)
    # tone-mark placement
    s = _TONE_RE.sub(lambda m: TONE_MAP[m.group(1)], s)
    # money shorthand must be handled before k -> không
    s = _MONEY_K.sub(r"\1 nghìn", s)
    s = _MONEY_TR.sub(r"\1 triệu", s)
    # elongation: 'lỗiiiii' -> 'lỗi', 'appppp' -> 'app', 'quáaaa' -> 'quá'  ('cccd' is protected: ccc != elongation)
    s = re.sub(r"\bcccd\b", "c_c_c_d", s)
    s = _TONED_ELONG.sub(lambda m: next(g for g in m.groups() if g), s)
    if _ELONG3.search(s):
        stats["has_elong"] += 1
        s = _ELONG3.sub(r"\1\1", s)          # 3+ -> 2; the vocabulary pass below decides 2 -> 1 ('app' keeps 'pp')
    s = s.replace("c_c_c_d", "cccd")
    s = _PUNCT_RUN.sub(r"\1", s)
    s = _DOTS.sub(" . ", s)
    # teencode (token level; protected tokens untouched)
    s = _SAIGON.sub("sài_", s)  # temporary guard for 'sài gòn'

    def repl(m: re.Match) -> str:
        w = m.group()
        if w in PROTECTED:
            return w
        new = TEENCODE.get(w)
        if new is None:
            return w
        stats["n_teencode"] += 1
        stats[f"tc::{w}"] += 1
        return new

    s = _WORD.sub(repl, s)
    s = s.replace("sài_", "sài")
    # space out punctuation (not inside numbers such as 1.5 or 10,000), collapse whitespace
    s = _SPACE_PUNCT.sub(lambda m: f" {m.group(0)} ", s)
    return re.sub(r"\s+", " ", s).strip()


def fix_residual_doubles(texts: pd.Series, min_count: int = 5) -> tuple[pd.Series, int]:
    """Second, data-driven pass for doubled letters left by the first pass ('lỗii', 'tốtt', 'okk', 'ngoon').
    A token is collapsed only if the collapsed form is frequent in the corpus (>= min_count) AND more
    frequent than the doubled form. This keeps legitimate doubles: 'app' stays because 'ap' is rare."""
    cnt = Counter(w for t in texts for w in t.split())
    fixed = 0

    def fix_tok(w: str) -> str:
        nonlocal fixed
        if w in PROTECTED or not _DOUBLE.search(w):
            return w
        c = _DOUBLE.sub(r"\1", w)
        if cnt.get(c, 0) >= min_count and cnt[c] > cnt.get(w, 0):
            fixed += 1
            return c
        return w

    return texts.map(lambda t: " ".join(fix_tok(w) for w in t.split())), fixed


# --------------------------------------------------------------------------- #
# 2. WORD SEGMENTATION
# --------------------------------------------------------------------------- #
def get_segmenter(name: str):
    if name == "underthesea":
        from underthesea import word_tokenize
        return lambda t: word_tokenize(t, format="text")
    if name == "pyvi":
        from pyvi import ViTokenizer
        return lambda t: ViTokenizer.tokenize(t)
    raise ValueError(name)


_PLACEHOLDERS = {"emojipos", "emojineg", "emojineu", "url", "email", "phone", "number"}
_PUNCT_ONLY = re.compile(r"^[\W\d_]+$")


def to_topic_tokens(seg: str) -> str:
    """Segmented text -> tokens for c-TF-IDF: drop punctuation, pure numbers and placeholders."""
    return " ".join(t for t in seg.split() if t not in _PLACEHOLDERS and not _PUNCT_ONLY.match(t.replace("_", "")))


def to_embed_text(clean: str) -> str:
    """Clean text for sentence-embedding models: drop placeholders; keep natural word order."""
    out = " ".join(t for t in clean.split() if t not in _PLACEHOLDERS)
    return re.sub(r"\s+([.,!?;:)])", r"\1", out).strip()


# --------------------------------------------------------------------------- #
# 2b. PROMOTIONAL SPAM  (found by inspecting the data: templated "TinChac" ads and obfuscated loan-site links)
# --------------------------------------------------------------------------- #
# These are almost all 5-star, are not about the app, and would be labelled Positive -> inflate positive
# sentiment and create a fake topic ("kiếm tiền, tinchac"). They are REMOVED (audit file kept). The rules are
# evaluated on the RAW text; the domain pattern is case-sensitive on purpose so that an ordinary sentence end
# such as "bó tay. com" is not caught. A second, weaker tier is only FLAGGED for manual review.
_SPAM_STRONG = re.compile(
    r"(?i:tinchac|\bc0m\b|\(\s*w\s*e\s*b\s*\)|\bktien\b|vaytotnhat|\bloan\s?\d{2,}"
    r"|tiếp thị liên kết|nhập\s+(?:ngay\s+)?mã(?:\s+giới thiệu)?\s*:?\s*[a-z]{0,4}\d{3,})"
    r"|\b(?:VAY|Vay)\d\b|\bK\s?(?:VAY|Vay)\b|\bKVAY\b"       # 'K VAY' brand: case-sensitive, 'k vay' = 'không vay' must survive
    r"|\b(?:[A-Z]{3,}\d*|[A-Za-z]*\d+[A-Za-z]*)\s*\.\s*(?:C0M|COM|VN|NET|VIP|c0m|com|vn|net|vip)\b")
_SPAM_ACTION = re.compile(r"(gõ|vào|truy cập|tìm|search)\s+(trang|web|website)|kiếm\s+(thêm|tiền)", re.I)
_SPAM_MONEY = re.compile(r"\b\d+\s*(?:tr|triệu|k)\b.*(?:nhận|kiếm|có ngay|0 lãi|miễn lãi)|nhận ngay\s*\d+", re.I)
_SPAM_SUSPECT = re.compile(r"(vào|truy cập|tìm|search)\s+(trang|web)|kiếm\s+(được\s+)?\d+\s*(?:tr|k|triệu)|mã giới thiệu", re.I)


def spam_reason(raw: str) -> str:
    raw = unicodedata.normalize("NFC", html.unescape(str(raw)))
    if _SPAM_STRONG.search(raw):
        return "strong_pattern"
    if _SPAM_ACTION.search(raw) and _SPAM_MONEY.search(raw):
        return "action+money"
    return ""


# --------------------------------------------------------------------------- #
# 3. PIPELINE
# --------------------------------------------------------------------------- #
def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")
    df = df[REQUIRED_COLS].copy()
    df.insert(0, "review_id", range(len(df)))          # stable id = original row order
    dropped: list[dict] = []

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        if mask.any():
            dropped.extend({"review_id": r, "reason": reason} for r in df.loc[mask, "review_id"])
            df = df.loc[~mask].copy()

    # --- missing values -------------------------------------------------- #
    df["Content"] = df["Content"].astype("string")
    drop(df["Content"].isna() | (df["Content"].str.strip() == ""), "empty_content")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    drop(~df["Score"].isin([1, 2, 3, 4, 5]), "invalid_score")
    df["Score"] = df["Score"].astype(int)
    df["Thumbs_Up"] = pd.to_numeric(df["Thumbs_Up"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    # NB: the App Store scraper returns no helpful votes, so Thumbs_Up = 0 for App Store rows means "not available"
    for c in ("App", "Platform"):
        df[c] = df[c].astype("string").str.strip()
    drop(df["App"].isna() | df["Platform"].isna(), "missing_app_or_platform")
    # --- dates: Google Play = naive 'YYYY-mm-dd HH:MM:SS', App Store = ISO-8601 with offset ---------- #
    # step1_merge.py already converted App Store UTC stamps to Vietnam time (naive strings, no offset). Naive values are
    # therefore read as local time; values that still carry 'Z' or an offset are converted to local time.
    raw_dt = df["Date"].astype(str).str.strip()
    has_tz = raw_dt.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if has_tz.any():
        dt[has_tz] = (pd.to_datetime(raw_dt[has_tz], format="ISO8601", utc=True, errors="coerce")
                      .dt.tz_convert(LOCAL_TZ).dt.tz_localize(None))
    if (~has_tz).any():
        dt[~has_tz] = pd.to_datetime(raw_dt[~has_tz], format="ISO8601", errors="coerce")
    drop(dt.isna(), "unparseable_date")
    df["Date"] = dt
    df = df.reset_index(drop=True)
    return df, dropped


def run(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw_n = None
    df, dropped = load_and_validate(Path(args.input))
    raw_n = len(df) + len(dropped)
    df["text_raw"] = df["Content"].astype(str)

    # true scrape duplicates only
    dup = df.duplicated(["App", "Platform", "Date", "text_raw"], keep="first")
    dropped.extend({"review_id": r, "reason": "exact_scrape_duplicate"} for r in df.loc[dup, "review_id"])
    df = df.loc[~dup].reset_index(drop=True)

    # promotional spam: removed from the analysis tables, kept in an audit file
    df["spam_reason"] = df["text_raw"].map(spam_reason)
    sp = df["spam_reason"] != ""
    df.loc[sp, ["review_id", "App", "Platform", "Date", "Score", "spam_reason", "text_raw"]].to_csv(
        out / "spam_removed.csv", index=False, encoding="utf-8-sig")
    dropped.extend({"review_id": r, "reason": "promotional_spam"} for r in df.loc[sp, "review_id"])
    spam_by = df.loc[sp].groupby(["App", "Score"]).size().rename("n").reset_index().to_dict("records")
    df = df.loc[~sp].reset_index(drop=True)
    sus = (df["Score"] >= 4) & df["text_raw"].str.contains(_SPAM_SUSPECT)
    df.loc[sus, ["review_id", "App", "Platform", "Score", "text_raw"]].to_csv(
        out / "spam_suspect_manual_review.csv", index=False, encoding="utf-8-sig")

    # --- cleaning -------------------------------------------------------- #
    global_stats: Counter = Counter()
    clean, row_flags = [], []
    for s in df["text_raw"]:
        st: Counter = Counter()
        clean.append(normalize_text(s, st))
        row_flags.append((st["has_emoji"] > 0, st["has_elong"] > 0, st["n_teencode"]))
        global_stats.update(st)
    df["text_clean"] = clean
    df["has_emoji"], df["has_elong"], df["n_teencode"] = zip(*row_flags)
    df["text_clean"], n_fixed = fix_residual_doubles(df["text_clean"])

    # rows that are empty after cleaning (e.g. only punctuation)
    empty = df["text_clean"].str.replace(r"[\W_]+", "", regex=True).eq("")
    dropped.extend({"review_id": r, "reason": "empty_after_cleaning"} for r in df.loc[empty, "review_id"])
    df = df.loc[~empty].reset_index(drop=True)

    # --- segmentation ---------------------------------------------------- #
    seg = get_segmenter(args.segmenter)
    df["text_seg"] = [seg(t) for t in df["text_clean"]]
    df["text_embed"] = df["text_clean"].map(to_embed_text)
    df["text_topic"] = df["text_seg"].map(to_topic_tokens)

    # --- features / flags ------------------------------------------------ #
    words = df["text_embed"].str.split().str.len().fillna(0).astype(int)
    df["n_words_clean"] = words
    df["n_chars_raw"] = df["text_raw"].str.len()
    df["is_unaccented"] = (~df["text_embed"].str.contains(VN_DIACRITICS)) & (words >= 3)
    # exact same long text (>=4 words) inside App x Platform = copy-paste/spam -> keep first only for topics
    long_dup = (words >= TOPIC_MIN_WORDS) & df.duplicated(["App", "Platform", "text_clean"], keep="first")
    df["is_dup_longtext"] = long_dup
    df["is_topic_eligible"] = (words >= TOPIC_MIN_WORDS) & ~long_dup
    df["year"] = df["Date"].dt.year
    df["quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    df["in_common_window"] = df["Date"] >= pd.Timestamp(COMMON_WINDOW_START)

    cols = ["review_id", "App", "Platform", "Date", "Score", "Thumbs_Up", "text_raw", "text_clean", "text_seg",
            "text_embed", "text_topic", "n_words_clean", "n_chars_raw", "n_teencode", "has_emoji", "has_elong",
            "is_unaccented", "is_dup_longtext", "is_topic_eligible", "year", "quarter", "year_month",
            "in_common_window"]
    df = df[cols]
    df.to_csv(out / "reviews_clean.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(out / "reviews_clean.parquet", index=False)
    except Exception as e:  # pyarrow not installed -> csv is enough
        print(f"[info] parquet skipped: {e}")

    # --- audit tables ----------------------------------------------------- #
    tc = pd.DataFrame([(k[4:], v) for k, v in global_stats.items() if k.startswith("tc::")],
                      columns=["token", "n_replaced"]).sort_values("n_replaced", ascending=False)
    tc["replacement"] = tc["token"].map(TEENCODE)
    tc.to_csv(out / "teencode_audit.csv", index=False, encoding="utf-8-sig")

    tok_cnt = Counter(w for t in df["text_clean"] for w in t.split())
    cand = pd.DataFrame([(w, c) for w, c in tok_cnt.items()
                         if len(w) <= 3 and c >= 10 and w not in TEENCODE and w not in PROTECTED
                         and w.isalpha() and not VN_DIACRITICS.search(w)],
                        columns=["token", "freq"]).sort_values("freq", ascending=False)
    cand.to_csv(out / "candidate_abbreviations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dropped).to_csv(out / "dropped_rows.csv", index=False)

    # --- report ------------------------------------------------------------ #
    report = {
        "rows_input": int(raw_n),
        "rows_dropped": {k: int(v) for k, v in Counter(d["reason"] for d in dropped).items()},
        "rows_output": int(len(df)),
        "rows_topic_eligible": int(df["is_topic_eligible"].sum()),
        "spam_removed_by_app_score": spam_by,
        "spam_suspect_flagged_for_manual_review": int(sus.sum()),
        "share_short_lt4_words": round(float((df["n_words_clean"] < TOPIC_MIN_WORDS).mean()), 4),
        "share_unaccented_3plus_words": round(float(df["is_unaccented"].mean()), 4),
        "share_with_emoji": round(float(df["has_emoji"].mean()), 4),
        "share_with_elongation": round(float(df["has_elong"].mean()), 4),
        "teencode_replacements": int(sum(v for k, v in global_stats.items() if k.startswith("tc::"))),
        "double_letter_fixes": int(n_fixed),
        "by_app_platform": df.groupby(["App", "Platform"]).size().rename("n").reset_index().to_dict("records"),
        "by_app_topic_eligible": df.groupby("App")["is_topic_eligible"].sum().astype(int).to_dict(),
        "date_min": str(df["Date"].min()), "date_max": str(df["Date"].max()),
        "common_window_start": COMMON_WINDOW_START,
        "n_in_common_window_by_app": df[df["in_common_window"]].groupby("App").size().to_dict(),
        "words_clean_describe": df["n_words_clean"].describe().round(2).to_dict(),
        "segmenter": args.segmenter,
    }
    (out / "preprocessing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


# --------------------------------------------------------------------------- #
# 4. SELF-TEST
# --------------------------------------------------------------------------- #
def self_test() -> None:
    cases = {
        "Lỗiiiii quá!!!!": "lỗi quá !",
        "ko đc vào app": "không được vào app",
        "gọi cskh k ai nghe": "gọi chăm sóc khách hàng không ai nghe",
        "phí 50k mỗi tháng": "phí 50 nghìn mỗi tháng",
        "khoá thẻ, thuỷ điện, quý khách": "khóa thẻ , thủy điện , quý khách",
        "tài khoản toán hoàn thoại huỳnh khoẻ": "tài khoản toán hoàn thoại huỳnh khỏe",
        "không đủ đk vay": "không đủ đk vay",
        "quét CCCD và cccd": "quét cccd và cccd",
        "appppp rất tệ": "app rất tệ",
        "quáaaa tệ okk": "quá tệ ok",
        "như *** tiền ứng": "như từ tục tiền ứng",
        "**đổi hình thức** nhận thẻ": "đổi hình thức nhận thẻ",
        "gọi 0912345678 hoặc a@b.com": "gọi phone hoặc email",
        "tệ bạc 🤣🤣🤣 😡": "tệ bạc emojineu emojineu emojineu emojineg",
        "sài gòn và sài app": "sài gòn và xài app",
        "ok okie oke": "ok ok ok",
        "mở_thẻ  _ngay": "mở thẻ ngay",
    }
    # filler lines give the vocabulary pass the corpus frequencies it needs
    filler = ["lỗi", "quá", "ok", "app", "tệ", "rất"] * 6
    first = [normalize_text(c) for c in cases] + filler
    second, _ = fix_residual_doubles(pd.Series(first))
    bad = 0
    for (src, exp), got in zip(cases.items(), second.iloc[: len(cases)]):
        ok = got == exp
        bad += not ok
        print(f"[{'OK ' if ok else 'FAIL'}] {src!r} -> {got!r}" + ("" if ok else f"   (expected {exp!r})"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_reviews_raw_merged.csv")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--segmenter", choices=["underthesea", "pyvi"], default="underthesea")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    self_test() if a.self_test else run(a)
