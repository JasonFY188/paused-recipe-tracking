"""
analysis/pause_resume/validation/make_sample.py
Builds the 100-sentence sample for checking Claude's waiting labels by hand. Reads only; changes no existing file.
Text is copied exactly as in HD_EPIC_Narrations.pkl / complete_recipes.json (not even whitespace is stripped).

Run (intention env): python analysis/pause_resume/validation/make_sample.py
Outputs (this folder): key_hidden.csv, label_me.xlsx
"""
import glob
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 0
N_PER_GROUP = 25
ROOT = Path(__file__).resolve().parents[3]
ANN = ROOT / "annotations"
PR = ROOT / "analysis" / "pause_resume"
OUT = PR / "validation"
VIDEO_ID_FIXES = {"P09-2024623-154547": "P09-20240623-154547"}      # same fix as pause_resume.py


def main():
    # the 2,395 labelled sentences: item -> narration_id, set, Claude's label
    key = pd.read_csv(PR / "tables" / "to_label_key.csv")
    lab = pd.read_csv(PR / "tables" / "claude_wait_labels.csv")
    lab = key.merge(lab[["item", "waiting"]], on="item", validate="1:1")

    # Step 1: 25 per (set, label) group, seed 0, then shuffle all 100
    parts, notes = [], []
    for s in ("before_gap", "baseline"):
        for w in (1, 0):
            g = lab[(lab.set == s) & (lab.waiting == w)]
            n = min(N_PER_GROUP, len(g))
            if n < N_PER_GROUP:
                notes.append(f"{s}/{'waiting' if w else 'not waiting'}: only {len(g)} available, all taken")
            parts.append(g.sample(n=n, random_state=SEED))
            print(f"{s:10s} {'waiting' if w else 'not waiting':12s} available {len(g):5d}  taken {n}")
    smp = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    smp["id"] = [f"S{i + 1:03d}" for i in range(len(smp))]

    # raw narrations (no stripping), ordered within each video for the before / after context
    d = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    d = d.sort_values(["video_id", "start_timestamp", "stop_timestamp"] if "stop_timestamp" in d else
                      ["video_id", "start_timestamp", "end_timestamp"]).reset_index(drop=True)
    d["prev"] = d.groupby("video_id").narration.shift(1)
    d["next"] = d.groupby("video_id").narration.shift(-1)
    nar = d.set_index("unique_narration_id")

    # recipe(s) of the activity blocks containing the narration's start (same rule as pause_resume.py)
    names = {k: v["name"] for k, v in json.load(open(ANN / "high-level" / "complete_recipes.json", encoding="utf-8")).items()}
    dur = pd.read_csv(ANN / "youtube-links" / "HD_EPIC_YouTube_URLs.csv").set_index("video_id").duration
    b = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ANN / "high-level" / "activities" / "*.csv")))],
                  ignore_index=True)
    b["video_id"] = b.video_id.replace(VIDEO_ID_FIXES)
    b["s"] = b.start_time.astype(float)
    b["e"] = pd.to_numeric(b.end_time, errors="coerce")
    b.loc[b.e.isna(), "e"] = b.video_id.map(dur)
    b = b[b.recipe_id.notna()]

    def recipes_of(nid):
        r = nar.loc[nid]
        g = b[(b.video_id == r.video_id) & (b.s <= float(r.start_timestamp)) & (float(r.start_timestamp) < b.e)]
        keys = sorted({f"{r.participant_id}_{x.strip()}" for ids in g.recipe_id for x in ids.split(",") if x.strip()})
        return " / ".join(names[k] for k in keys)

    smp["participant"] = smp.narration_id.map(nar.participant_id)
    smp["recipe"] = smp.narration_id.map(recipes_of)
    smp["before"] = smp.narration_id.map(nar.prev).fillna("(start of video)")
    smp["sentence"] = smp.narration_id.map(nar.narration)
    smp["after"] = smp.narration_id.map(nar.next).fillna("(end of video)")
    assert smp.sentence.notna().all() and (smp.recipe != "").all()

    # hidden key
    smp.rename(columns={"item": "original_item", "waiting": "claude_label"})[
        ["id", "original_item", "narration_id", "set", "claude_label"]].to_csv(OUT / "key_hidden.csv", index=False)

    # Step 2: the sheet to label (no labels, no set)
    wb = Workbook()
    ws = wb.active
    ws.title = "label_me"
    head = ["id", "participant", "recipe (as written)", "sentence BEFORE (as written)",
            "SENTENCE TO LABEL (as written)", "sentence AFTER (as written)", "my_label", "my_note"]
    ws.append(head)
    for r in smp.itertuples():
        ws.append([r.id, r.participant, r.recipe, r.before, r.sentence, r.after, None, None])
    widths = [7, 11, 24, 45, 55, 45, 14, 40]
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    grey, bold = Font(color="808080"), Font(bold=True)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="404040")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row[3].font = row[5].font = grey
        row[4].font = Font(bold=True, size=12)
        row[6].fill = row[7].fill = PatternFill("solid", fgColor="FFF8DC")
    dv = DataValidation(type="list", formula1='"waiting,not waiting,unsure"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Choose one", error="Choose: waiting / not waiting / unsure")
    ws.add_data_validation(dv)
    dv.add(f"G2:G{len(smp) + 1}")
    ws.freeze_panes = "A2"
    wb.save(OUT / "label_me.xlsx")

    print(f"{len(smp)} sentences -> label_me.xlsx, key_hidden.csv")
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
