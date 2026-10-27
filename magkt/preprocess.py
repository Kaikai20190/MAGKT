"""Build sequence files and the question-concept hypergraph H from raw data.

One pipeline for every dataset, so that question ids, concept ids and H are produced from
the same mapping and cannot drift apart:

  raw table -> clean -> order each learner's interactions in time ->
  map questions to 1..Q and concepts to 1..C -> split learners 80/20 ->
  write  <name>_train.txt / <name>_test.txt  and  H/<name>.csv  and  <name>_stats.json

Output format (4 lines per learner, the format of the preprocessed ASSISTments2017 files):
    length
    question ids (1-based)
    concept id of each interaction (1-based; the first concept for multi-concept questions)
    responses (1 correct / 0 incorrect)
  and, if timestamps are available, a 5th line with the time in seconds.

H (2Q x 2C, no header):
    row q-1     (correct node of question q)    has a 1 in column c-1      for every concept c of q
    row Q+q-1   (incorrect node of question q)  has a 1 in column C+c-1
  i.e. every concept is split into a "correct" and an "incorrect" hyperedge.
"""
import json
import os

import numpy as np
import pandas as pd


# ------------------------------------------------------------ helpers
def find_col(df, candidates, required=True, what=""):
    cols = list(df.columns)
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    if required:
        raise KeyError(f"No {what} column among {candidates}. Columns in the file: {cols}")
    return None


def _to_seconds(series):
    t = pd.to_datetime(series, errors="coerce")
    return (t - pd.Timestamp("1970-01-01")).dt.total_seconds()


def build(df, name, out_dir, user, question, concepts, correct, order=None, time=None,
          test_ratio=0.2, split_seed=42, min_len=3, missing_kc="drop", log=print):
    """df columns: user, question (str key), concepts (list of str), correct (0/1), optional order/time.

    missing_kc: "drop" removes interactions without a concept; "keep" keeps them (concept id 0 on
    the concept line; a question that never has a concept gets an empty row in H)."""
    stats = {"dataset": name, "rows_in": int(len(df)), "missing_kc": missing_kc}
    df = df[[c for c in [user, question, concepts, correct, order, time] if c is not None]].copy()
    df = df.dropna(subset=[user, question, correct])
    stats["rows_valid_response"] = int(len(df))
    has_kc = df[concepts].map(len) > 0
    stats["rows_without_concept"] = int((~has_kc).sum())
    stats["questions_before_concept_filter"] = int(df[question].nunique())
    if missing_kc == "drop":
        df = df[has_kc]
    elif missing_kc != "keep":
        raise ValueError(missing_kc)
    stats["rows_after_concept_filter"] = int(len(df))

    sort_cols = [c for c in (time, order) if c is not None]
    df = df.sort_values([user] + sort_cols, kind="stable")
    counts = df.groupby(user)[question].transform("size")
    df = df[counts >= min_len]
    stats["rows_after_min_len"] = int(len(df))

    # ---- id maps (sorted keys -> reproducible)
    q_keys = sorted(df[question].astype(str).unique())
    q_map = {k: i + 1 for i, k in enumerate(q_keys)}
    c_keys = sorted({c for cs in df[concepts] for c in cs})
    c_map = {k: i + 1 for i, k in enumerate(c_keys)}
    Q, C = len(q_keys), len(c_keys)
    df["_q"] = df[question].astype(str).map(q_map)
    df["_cs"] = df[concepts].map(lambda cs: sorted(c_map[c] for c in cs))
    df["_c"] = df["_cs"].map(lambda cs: cs[0] if cs else 0)

    # ---- question -> concepts (union over all occurrences)
    q_concepts = {}
    for q, cs in zip(df["_q"].values, df["_cs"].values):
        q_concepts.setdefault(int(q), set()).update(cs)
    n_multi = sum(len(v) > 1 for v in q_concepts.values())

    H = np.zeros((2 * Q, 2 * C), dtype=np.int8)
    for q, cs in q_concepts.items():
        for c in cs:
            H[q - 1, c - 1] = 1
            H[Q + q - 1, C + c - 1] = 1

    # ---- learner split
    users = np.array(sorted(df[user].astype(str).unique().tolist()), dtype=object)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(users)
    n_test = int(round(test_ratio * len(users)))
    test_users = set(users[:n_test].tolist())

    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "H"), exist_ok=True)
    has_t = time is not None and df[time].notna().all()
    files = {"train": open(os.path.join(out_dir, name, f"{name}_train.txt"), "w"),
             "test": open(os.path.join(out_dir, name, f"{name}_test.txt"), "w")}
    n_learn = {"train": 0, "test": 0}
    n_int = {"train": 0, "test": 0}
    for u, g in df.groupby(user, sort=False):
        part = "test" if str(u) in test_users else "train"
        f = files[part]
        f.write(f"{len(g)}\n")
        f.write(",".join(map(str, g["_q"].values)) + "\n")
        f.write(",".join(map(str, g["_c"].values)) + "\n")
        f.write(",".join(map(str, g[correct].astype(int).values)) + "\n")
        if has_t:
            f.write(",".join(str(int(x)) for x in g[time].values) + "\n")
        n_learn[part] += 1
        n_int[part] += len(g)
    for f in files.values():
        f.close()
    np.savetxt(os.path.join(out_dir, "H", f"{name}.csv"), H, fmt="%d", delimiter=",")

    stats.update({
        "learners": int(len(users)), "learners_train": n_learn["train"], "learners_test": n_learn["test"],
        "questions": Q, "concepts": C, "interactions": int(len(df)),
        "interactions_train": n_int["train"], "interactions_test": n_int["test"],
        "questions_with_multiple_concepts": int(n_multi),
        "questions_without_concept": int(Q - sum(len(v) > 0 for v in q_concepts.values())),
        "concepts_per_question_mean": float(np.mean([len(v) for v in q_concepts.values()])) if q_concepts else 0.0,
        "interactions_with_multiple_concepts": int((df["_cs"].map(len) > 1).sum()),
        "positive_rate": float(df[correct].mean()),
        "has_timestamps": bool(has_t), "n_lines": 5 if has_t else 4,
        "test_ratio": test_ratio, "split_seed": split_seed, "min_len": min_len,
    })
    with open(os.path.join(out_dir, name, f"{name}_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    pd.Series(q_map).rename_axis("raw_question").rename("question_id").to_csv(
        os.path.join(out_dir, name, f"{name}_question_map.csv"))
    pd.Series(c_map).rename_axis("raw_concept").rename("concept_id").to_csv(
        os.path.join(out_dir, name, f"{name}_concept_map.csv"))
    log(f"[{name}] raw rows {stats['rows_in']}, valid responses {stats['rows_valid_response']}, rows without "
        f"concept {stats['rows_without_concept']} ({'removed' if missing_kc == 'drop' else 'kept'}), "
        f"interactions after removing short sequences {stats['interactions']}")
    log(f"[{name}] learners {stats['learners']} (train {n_learn['train']} / test {n_learn['test']}), "
        f"Q = {Q}, C = {C}, interactions {stats['interactions']}, "
        f"multi-concept questions {n_multi}, questions without concept {stats['questions_without_concept']}, "
        f"concepts per question {stats['concepts_per_question_mean']:.2f}, "
        f"correct rate {stats['positive_rate']:.3f}, timestamps {'yes' if has_t else 'no'}")
    return stats


def _split_kc(series, sep):
    return series.fillna("").astype(str).map(lambda s: [k.strip() for k in s.split(sep) if k.strip()])


# --------------------------------------------------------- datasets
def prep_assist2009(path, out_dir, name="assist2009", granularity="problem", **kw):
    """ASSISTments 2009-2010 skill builder (skill_builder_data*.csv).

    granularity = "problem": question = problem_id, concept = skill_id.
    granularity = "skill"  : question = the skill (combination) of the interaction, e.g. "12" or
                             "12_45" for a two-skill answer (classic DKT setting, but multi-skill
                             answers stay ONE interaction instead of being duplicated per skill,
                             which would leak the label); concepts = the atomic skills. Multi-skill answers appear as several rows with
    the same order_id; they are merged into one interaction with several concepts.
    Order: order_id (no reliable timestamps in this file)."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    u = find_col(df, ["user_id"], what="learner")
    q = find_col(df, ["problem_id"], what="question")
    s = find_col(df, ["skill_id"], what="concept")
    c = find_col(df, ["correct"], what="response")
    o = find_col(df, ["order_id"], what="order")
    df = df[df[c].isin([0, 1])]
    df = df.dropna(subset=[s])
    df[s] = df[s].astype(str).str.replace(r"\.0$", "", regex=True)
    g = df.groupby(o, sort=False).agg({u: "first", q: "first", c: "first", s: lambda x: sorted(set(x))})
    g = g.reset_index()
    g.columns = ["order", "user", "question", "correct", "concepts"]
    g["question"] = g["question"].astype(str)
    if granularity == "skill":
        g["question"] = g["concepts"].map(lambda cs: "_".join(cs))
    elif granularity != "problem":
        raise ValueError(granularity)
    return build(g, name, out_dir, "user", "question", "concepts", "correct", order="order", **kw)


def _datashop(df, name, out_dir, correct_col, correct_map, kc_col, granularity, kc_sep="~~",
              hierarchy=False, **kw):
    u = find_col(df, ["Anon Student Id"], what="learner")
    pn = find_col(df, ["Problem Name"], what="problem name")
    sn = find_col(df, ["Step Name"], what="step name", required=(granularity == "problem_step"))
    ph = find_col(df, ["Problem Hierarchy"], required=False)
    kc = find_col(df, kc_col, what="concept")
    cc = find_col(df, correct_col, what="response")
    tt = find_col(df, ["First Transaction Time", "Step Start Time"], required=False)
    rw = find_col(df, ["Row"], required=False)
    kc_cols = [c for c in df.columns if c.strip().upper().startswith("KC")]
    if kc_cols:
        print(f"[{name}] concept columns in the file (share non-empty / distinct values):")
        for c in kc_cols:
            nn = df[c].notna() & (df[c].astype(str).str.strip() != "")
            print(f"      {c:30s} {nn.mean():.3f}  {df.loc[nn, c].nunique()}")
    print(f"[{name}] columns used: learner={u}, question={pn}" + (f"+{sn}" if granularity == "problem_step" else "")
          + f", concept={kc}, response={cc}, time={tt}, order={rw}")

    out = pd.DataFrame({"user": df[u].astype(str)})
    key = df[pn].astype(str)
    if hierarchy and ph is not None:
        key = df[ph].astype(str) + "|" + key
    if granularity == "problem_step":
        key = key + "|" + df[sn].astype(str)
    out["question"] = key
    out["concepts"] = _split_kc(df[kc], kc_sep)
    raw = df[cc]
    out["correct"] = raw.map(correct_map) if correct_map else pd.to_numeric(raw, errors="coerce")
    out = out[out["correct"].isin([0, 1])]
    out["order"] = pd.to_numeric(df.loc[out.index, rw], errors="coerce") if rw else np.arange(len(out))
    out["time"] = _to_seconds(df.loc[out.index, tt]) if tt else np.nan
    time_col = "time" if tt and out["time"].notna().all() else None
    return build(out, name, out_dir, "user", "question", "concepts", "correct", order="order", time=time_col, **kw)


def prep_statics2011(path, out_dir, name="static2011", granularity="problem_step", hint_as_incorrect=True,
                     kc_col=("KC (F2011)", "KC(F2011)", "KC (Single-KC)", "KC"), **kw):
    """OLI Engineering Statics Fall 2011, DataShop student-step export (AllData_student_step_2011F.csv).

    question = Problem Name + Step Name (the usual 'Statics2011' definition), concept = KC (F2011).
    correct = First Attempt == 'correct'; 'hint' counts as incorrect unless hint_as_incorrect=False
    (then hint rows are removed)."""
    sep = "\t" if path.endswith((".txt", ".tsv")) else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    cmap = {"correct": 1, "incorrect": 0}
    if hint_as_incorrect:
        cmap["hint"] = 0
    fa = find_col(df, ["First Attempt"], what="first attempt")
    df[fa] = df[fa].astype(str).str.strip().str.lower()
    return _datashop(df, name, out_dir, [fa], cmap, list(kc_col), granularity, **kw)


def prep_kddcup2010(path, out_dir, name="kddcup2010", granularity="problem",
                    kc_col=("KC(Default)", "KC (Default)", "KC(SubSkills)", "KC"), **kw):
    """KDD Cup 2010 (e.g. algebra_2005_2006_train.txt, tab separated).

    question = Problem Name (granularity='problem', default) or Problem Name + Step Name
    ('problem_step', very many questions); concept = KC(Default); correct = Correct First Attempt.
    Rows without a KC are removed (standard practice)."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    return _datashop(df, name, out_dir, ["Correct First Attempt"], None, list(kc_col), granularity, **kw)
