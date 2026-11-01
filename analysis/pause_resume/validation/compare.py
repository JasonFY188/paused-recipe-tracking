"""
analysis/pause_resume/validation/compare.py
Compares the user's hand labels (label_me.xlsx) with Claude's labels (key_hidden.csv). Changes no existing file.

The event-level "what if" needs the events and the 400 baseline windows. They are rebuilt by running
pause_resume.main() with every output folder redirected to a temporary folder (nothing in analysis/pause_resume/
is overwritten), then checked against the published numbers (26.8% / 6.0%).

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/pause_resume/validation/compare.py
Output: validation/compare_results.json (validation_report.md is written by hand from it)
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
N_DRAWS = 20000
HERE = Path(__file__).resolve().parent
PR = HERE.parent
POP = {("before_gap", 1): 146, ("before_gap", 0): 1075, ("baseline", 1): 30, ("baseline", 0): 1144}   # all 2,395 labels
# Scenario C: rows labelled "not waiting" whose note says the action leads to / induces waiting (Claude's reading of the notes)
NOTE_SAYS_LEADS_TO_WAITING = ["S014", "S015", "S018", "S020"]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def jeffreys(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(.025, k + .5, n - k + .5)) if k > 0 else 0.0,
            float(beta.ppf(.975, k + .5, n - k + .5)) if k < n else 1.0]


def rebuild_events():
    spec = importlib.util.spec_from_file_location("pause_resume", PR / "pause_resume.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    tmp = Path(tempfile.mkdtemp(prefix="pr_rebuild_"))
    pr.OUT, pr.TAB, pr.FIG = tmp, tmp / "tables", tmp / "figures"          # LABELS still points to the real (read-only) file
    grab = {}

    def prof(frame, event, arg):
        if event == "return" and frame.f_code is pr.main.__code__:
            grab.update(frame.f_locals)
    sys.setprofile(prof)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pr.main()
    finally:
        sys.setprofile(None)
    same_key = pd.read_csv(tmp / "tables" / "to_label_key.csv").equals(pd.read_csv(PR / "tables" / "to_label_key.csv"))
    ev, base_win, cw = grab["ev"], grab["base_win"], grab["cw"]
    ev_kn = np.array([(sum(cw(i) for i in l), len(l)) for l in ev.last3])
    win_kn = np.array([(sum(cw(i) for i in w), len(w)) for _, w in base_win])
    return ev_kn, win_kn, same_key


def rate(kn, ppv, fpr_miss):
    """share of events/windows with >= 1 waiting narration if each Claude-waiting one is truly waiting with prob ppv
    and each Claude-not-waiting one is truly waiting with prob fpr_miss (vectorised over draws)"""
    k, n = kn[:, 0][None, :], kn[:, 1][None, :]
    ppv, miss = np.atleast_1d(ppv)[:, None], np.atleast_1d(fpr_miss)[:, None]
    return 100 * (1 - (1 - ppv) ** k * (1 - miss) ** (n - k)).mean(1)


def _ab(d, st):
    """Jeffreys Beta parameters of P(me waiting | Claude waiting) in one set"""
    ds = d[(d.set == st) & (d.claude_label == 1)]
    return int(ds.u.sum()) + .5, len(ds) - int(ds.u.sum()) + .5


def pc0(v):
    return [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]


def main():
    x = pd.read_excel(HERE / "label_me.xlsx")
    key = pd.read_csv(HERE / "key_hidden.csv")
    m = key.merge(x[["id", "my_label", "my_note"]], on="id")
    m = m[m.my_label.notna()].copy()
    m["my_note"] = m.my_note.fillna("")
    text = dict(zip(x.id, x["SENTENCE TO LABEL (as written)"]))
    res = dict(n_labelled=len(m), my_label_counts=m.my_label.value_counts().to_dict(),
               labelled_per_group={f"{s}/{c}": int(n) for (s, c), n in m.groupby(["set", "claude_label"]).size().items()})

    ev_kn, win_kn, same_key = rebuild_events()
    pub_ev, pub_win = rate(ev_kn, 1.0, 0.0)[0], rate(win_kn, 1.0, 0.0)[0]
    res["rebuild_check"] = dict(same_label_key=same_key, events=len(ev_kn), windows=len(win_kn),
                                event_pct=round(pub_ev, 1), baseline_pct=round(pub_win, 1))
    assert same_key and round(pub_ev, 1) == 26.8 and round(pub_win, 1) == 6.0

    scen = {"A_as_labelled": m.my_label.map({"waiting": 1, "not waiting": 0}),
            "B_unsure_as_waiting": m.my_label.map({"waiting": 1, "not waiting": 0, "unsure": 1}),
            "C_notes_leads_to_waiting": m.my_label.map({"waiting": 1, "not waiting": 0})
                                         .where(~m.id.isin(NOTE_SAYS_LEADS_TO_WAITING), 1)}
    rng = np.random.default_rng(SEED)
    res["scenarios"] = {}
    for name, u in scen.items():
        d = m.assign(u=u)[u.notna()]
        c, y = d.claude_label.values, d.u.astype(int).values
        tab = pd.crosstab(pd.Series(y, name="me"), pd.Series(c, name="claude")).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
        # population-weighted recall (the sample has 50% Claude-waiting; the 2,395 sentences have 7.3%)
        w = d.apply(lambda r: POP[(r.set, r.claude_label)] / ((d.set == r.set) & (d.claude_label == r.claude_label)).sum(), axis=1).values
        s = dict(n_used=len(d), agreement_pct=round(100 * (c == y).mean(), 1), kappa=round(kappa(c, y), 2),
                 table_me_rows_claude_cols={"me_wait": tab.loc[1].tolist(), "me_not": tab.loc[0].tolist()},
                 precision_pct=round(100 * y[c == 1].mean(), 1), precision_ci=jeffreys(int(y[c == 1].sum()), int((c == 1).sum())),
                 recall_raw_pct=round(100 * c[y == 1].mean(), 1) if (y == 1).any() else None,
                 recall_weighted_pct=round(100 * (w * c * y).sum() / (w * y).sum(), 1) if (y == 1).any() else None,
                 claude_not_but_me_wait=f"{int(y[c == 0].sum())}/{int((c == 0).sum())}",
                 claude_not_but_me_wait_ci=jeffreys(int(y[c == 0].sum()), int((c == 0).sum())))
        s["per_set"] = {}
        for st in ("before_gap", "baseline"):
            ds = d[d.set == st]
            cs, ys = ds.claude_label.values, ds.u.astype(int).values
            s["per_set"][st] = dict(n=len(ds), agreement_pct=round(100 * (cs == ys).mean(), 1), kappa=round(kappa(cs, ys), 2),
                                    precision=f"{int(ys[cs == 1].sum())}/{int((cs == 1).sum())}",
                                    claude_not_but_me_wait=f"{int(ys[cs == 0].sum())}/{int((cs == 0).sum())}")
        # what-if: PPV per set; miss rate pooled over both sets (0 or few cases each, no sign it differs)
        miss_k, miss_n = int(y[c == 0].sum()), int((c == 0).sum())
        point, draws = {}, {}
        miss_draw = rng.beta(miss_k + .5, miss_n - miss_k + .5, N_DRAWS)        # one shared draw for both sets
        for st, kn in (("before_gap", ev_kn), ("baseline", win_kn)):
            ds = d[(d.set == st) & (d.claude_label == 1)]
            k1, n1 = int(ds.u.sum()), len(ds)
            point[st] = rate(kn, k1 / n1, miss_k / miss_n)[0]
            draws[st] = rate(kn, rng.beta(k1 + .5, n1 - k1 + .5, N_DRAWS), miss_draw)
        s["what_if_if_claude_misses_none"] = dict(
            events_pct=round(float(np.median(rate(ev_kn, rng.beta(*_ab(d, "before_gap"), N_DRAWS), 0.0))), 1),
            events_ci=pc0(rate(ev_kn, rng.beta(*_ab(d, "before_gap"), N_DRAWS), 0.0)),
            baseline_ci=pc0(rate(win_kn, rng.beta(*_ab(d, "baseline"), N_DRAWS), 0.0)))
        pc = lambda v: [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]
        diff, ratio = draws["before_gap"] - draws["baseline"], draws["before_gap"] / draws["baseline"]
        s["what_if"] = dict(events_pct=round(point["before_gap"], 1), events_ci=pc(draws["before_gap"]),
                            baseline_pct=round(point["baseline"], 1), baseline_ci=pc(draws["baseline"]),
                            diff_points=round(point["before_gap"] - point["baseline"], 1), diff_ci=pc(diff),
                            ratio=round(point["before_gap"] / point["baseline"], 1) if point["baseline"] else None,
                            ratio_ci=pc(ratio), p_diff_gt_0=round(float((diff > 0).mean()), 3))
        res["scenarios"][name] = s

    dis = m.assign(u=scen["A_as_labelled"])
    dis = dis[dis.u.isna() | (dis.u != dis.claude_label)]
    res["disagreements"] = [dict(id=r.id, set=r.set, sentence=text[r.id], me=r.my_label,
                                 claude="waiting" if r.claude_label else "not waiting", note=r.my_note)
                            for r in dis.itertuples()]
    json.dump(res, open(HERE / "compare_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1, ensure_ascii=False))
    print(len(res["disagreements"]), "disagreements / unsure")


if __name__ == "__main__":
    main()
