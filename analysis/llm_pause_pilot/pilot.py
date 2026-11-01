"""
analysis/llm_pause_pilot/pilot.py
Can a local LLM keep track of a paused recipe? Unattended, checkpointed run (see report.md for all decisions).

Model      qwen3:8b through the local Ollama server; temperature 0, seed 0, thinking off, num_ctx 40960 (the model's
           maximum), JSON output constrained by a schema
Input      the person's raw HD-EPIC narrations of the current video, oldest first, each as "[mm:ss] narration";
           FULL = since the video start (oldest lines dropped only if over HIST_CHARS), LAST10 = the last 10
Output     per recipe: state (active / paused / finished / not started) + confidence; next recipe + confidence
Order      events shuffled with seed 0; for each event: A-full, A-last10, B-full, B-last10
Checkpoint every answer is appended to results.jsonl; a restart skips (case, condition) pairs already there
Time limit 7 hours per run; afterwards the run stops cleanly and analyze.py writes the report

Run:  python analysis/llm_pause_pilot/pilot.py [--test 5] [--hours 7]
"""
import argparse
import ctypes
import json
import pathlib
import random
import subprocess
import sys
import time
import urllib.request

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
STATUS = HERE / "status.json"
URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
NUM_CTX = 40960
HIST_CHARS = 110_000                         # about 30k tokens of history; the rest of the context is prompt + answer
SEED = 0
STATES = ["active", "paused", "finished", "not started"]

SYSTEM = (
    "You watch a person working in their own home kitchen. You receive what the person did, one action per line, "
    "oldest first, each with its time in the video [minutes:seconds]. You also receive the list of recipes this person "
    "makes. The person may work on several recipes, leave a recipe for a while (for example to wash up, tidy, or work "
    "on another dish) and come back to it later.\n"
    "For EVERY recipe in the list, give its state at the moment of the LAST line:\n"
    "- active: the person is working on this recipe right now (at the last line)\n"
    "- paused: the recipe was started and is not finished, but the person is doing something else right now\n"
    "- finished: the recipe was completed earlier\n"
    "- not started: no step of this recipe has been done yet in what you see\n"
    "Give a confidence between 0 and 1 for each state. Then say which recipe the person will work on next "
    "(the recipe they continue or start next; it may be the active one), or \"none\", with a confidence.\n"
    "Answer only with JSON.")


def schema(ids):
    # "evidence" (time of the last line belonging to the recipe) comes before the state: in the 5-case test the plain
    # format copied one state to every recipe; with the evidence field the states differed by recipe (report.md)
    return {"type": "object", "properties": {
        "recipes": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string", "enum": ids}, "evidence": {"type": "string"},
            "state": {"type": "string", "enum": STATES},
            "confidence": {"type": "number"}}, "required": ["id", "evidence", "state", "confidence"]}},
        "next_recipe": {"type": "string", "enum": ids + ["none"]}, "next_confidence": {"type": "number"}},
        "required": ["recipes", "next_recipe", "next_confidence"]}


def fmt(t):
    return f"[{int(t // 60):02d}:{int(t % 60):02d}]"


def build_prompt(case, condition):
    hist = case.history if condition == "full" else case.history[-10:]
    lines = [f"{fmt(t)} {x}" for t, x in hist]
    dropped = 0
    while sum(len(l) + 1 for l in lines) > HIST_CHARS:              # keep the most recent lines that fit
        lines.pop(0); dropped += 1
    recipes = "\n".join(f"{k}: {v}" for k, v in case.recipe_list.items())
    head = "(only the most recent actions are shown)\n" if condition == "last10" or dropped else ""
    user = f"Recipes of this person:\n{recipes}\n\nActions so far:\n{head}" + "\n".join(lines) + \
           "\n\nGive the state of every recipe at the last line, and the next recipe, as JSON. For each recipe first " \
           "write as evidence the time of the last line that belongs to it (or \"none\"), then its state."
    return user, len(hist) - dropped, dropped


def ask(user, ids):
    body = dict(model=MODEL, stream=False, think=False, format=schema(ids),
                options=dict(temperature=0, seed=SEED, num_ctx=NUM_CTX),
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def done_keys():
    if not RESULTS.exists():
        return set()
    keys = set()
    for l in open(RESULTS, encoding="utf-8"):
        try:
            d = json.loads(l)
            keys.add((d["case"], d["condition"]))
        except json.JSONDecodeError:
            pass                                                     # a half-written last line after a crash
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="run only this many cases (full + last10) and print them")
    ap.add_argument("--hours", type=float, default=7.0)
    args = ap.parse_args()
    # keep Windows from sleeping while this process runs (released automatically when it exits)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    cases = pd.read_pickle(HERE / "data" / "cases.pkl")
    events = sorted(cases.event_id.unique())
    random.Random(SEED).shuffle(events)
    order = {e: i for i, e in enumerate(events)}
    cases = cases.assign(_o=cases.event_id.map(order)).sort_values(["_o", "point"])
    if args.test:
        cases = cases.head(args.test)
        out_file = HERE / "test_outputs.jsonl"
    else:
        out_file = RESULTS
    done = set() if args.test else done_keys()
    t_start, n_new = time.time(), 0
    stopped = "finished"
    with open(out_file, "a" if not args.test else "w", encoding="utf-8") as f:
        for c in cases.itertuples():
            for cond in ("full", "last10"):
                if (c.case, cond) in done:
                    continue
                if time.time() - t_start > args.hours * 3600:
                    stopped = "time limit"
                    break
                ids = list(c.recipe_list)
                user, n_shown, dropped = build_prompt(c, cond)
                rec = dict(case=c.case, condition=cond, event_id=str(c.event_id), point=c.point, n_shown=n_shown,
                           n_dropped=dropped)
                try:
                    out, secs = ask(user, ids)
                    rec.update(raw=out["message"]["content"], seconds=round(secs, 2),
                               prompt_tokens=out.get("prompt_eval_count"), answer_tokens=out.get("eval_count"))
                    try:
                        rec["answer"] = json.loads(out["message"]["content"])
                    except json.JSONDecodeError:
                        rec["answer"] = None
                except Exception as ex:                              # server error: record and continue
                    rec.update(error=repr(ex), seconds=None, answer=None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                n_new += 1
                if args.test:
                    print(f"--- {c.case} [{cond}] R = {c.recipe_id} ({c.recipe_name}), shown {n_shown} lines, {rec.get('seconds')} s, "
                          f"prompt tokens {rec.get('prompt_tokens')}")
                    print(json.dumps(rec.get("answer"), ensure_ascii=False))
                else:
                    json.dump(dict(state="running", done=len(done) + n_new, total=2 * len(cases),
                                   hours=round((time.time() - t_start) / 3600, 2)), open(STATUS, "w"))
            else:
                continue
            break
    if not args.test:
        json.dump(dict(state=stopped, done=len(done_keys()), total=2 * len(cases),
                       hours=round((time.time() - t_start) / 3600, 2), ended=time.strftime("%Y-%m-%d %H:%M")),
                  open(STATUS, "w"))
        subprocess.run([sys.executable, str(HERE / "analyze.py")], check=False)


if __name__ == "__main__":
    main()
