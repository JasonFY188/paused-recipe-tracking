"""
analysis/recipe_groups/step1_propose_kinds.py
Claude's PROPOSED split of HD-EPIC noun classes into 5 kinds (tool / ingredient / fixture / cleaning / other), for the
user to confirm or edit. Whole categories where possible; every class moved against its category is listed with a
reason. Writes tables/step1_proposed_kinds.csv (edit its "kind" column to change the split) and the coverage per kind.

Run (intention env): python analysis/recipe_groups/step1_propose_kinds.py
"""
import pathlib

import pandas as pd

TAB = pathlib.Path(__file__).resolve().parent / "tables"

CATEGORY_KIND = {
    "utensils": "tool", "cookware": "tool", "crockery": "tool", "cutlery": "tool", "containers": "tool",
    "materials": "tool", "appliances": "fixture", "furniture": "fixture", "storage": "fixture",
    "cleaning equipment and material": "cleaning",
    "vegetables": "ingredient", "fruits and nuts": "ingredient", "spices and herbs and sauces": "ingredient",
    "baked goods and grains": "ingredient", "prepared food": "ingredient", "meat and substitute": "ingredient",
    "dairy and eggs": "ingredient", "drinks": "ingredient",
    "rubbish": "other", "other": "other", "hand": "other",
}
# class -> (kind, reason); only classes whose kind differs from their category's kind
MOVES = {
    # appliances: default fixture; movable ones are tools, non-kitchen devices are other
    "scale": ("tool", "movable kitchen scale"), "kettle": ("tool", "movable appliance"),
    "blender": ("tool", "movable appliance"), "maker:coffee": ("tool", "movable appliance"),
    "processor:food": ("tool", "movable appliance"), "toaster": ("tool", "movable appliance"),
    "cooker:slow": ("tool", "movable appliance (slow / pressure cooker)"),
    "machine:hot:chocolate": ("tool", "movable appliance"), "machine:sous:vide": ("tool", "movable appliance"),
    "machine:candy:floss": ("tool", "movable appliance"),
    "phone": ("other", "not a kitchen item"), "tv": ("other", "not a kitchen item"), "camera": ("other", "not a kitchen item"),
    "watch": ("other", "not a kitchen item"), "computer": ("other", "not a kitchen item"),
    "battery": ("other", "not a kitchen item"), "control:remote": ("other", "not a kitchen item"),
    "cd": ("other", "not a kitchen item"), "alarm": ("other", "timer / time, mostly phone or oven clock"),
    "heat": ("other", "a setting (temperature), not an object"),
    "hoover": ("cleaning", "vacuum cleaner"),
    # materials: default tool (foil, cling film, baking sheets, napkins, gloves, mats)
    "wrap": ("ingredient", "instances are mostly tortilla / taco / kebab wrap (food)"),
    "paper": ("other", "mostly notebook / letters / receipts (baking paper also inside)"),
    "clothes": ("other", "laundry"), "label": ("other", "not fetched"), "rubber": ("other", "rubber bands / feet"),
    "apron": ("other", "worn, not fetched"), "pillow": ("other", "not a kitchen item"),
    "towel": ("cleaning", "tea towel / hand towel"), "towel:kitchen": ("cleaning", "tea towel"),
    # containers / utensils / cookware: default tool
    "backpack": ("other", "not a kitchen item"), "pen": ("other", "writing"), "key": ("other", "not a kitchen item"),
    "tape": ("other", "not a kitchen item"), "holder": ("other", "knife block / holders, mostly fixed"),
    "handle": ("other", "a part of anything (pan, drawer, oven, tap)"),
    # cleaning: default cleaning
    "tablet": ("other", "dishwasher tablet mixed with rennet tablet and e-reader"),
    # furniture / storage: default fixture
    "candle": ("other", "not a fixture"),
    # rubbish: default other; the bin stays where it is
    "bin": ("fixture", "bin, not fetched"),
    # other: default other
    "ice": ("ingredient", "ice cubes are an ingredient"),
}


def main():
    c = pd.read_csv(TAB / "step1_classes.csv")
    c["category_kind"] = c.category.map(CATEGORY_KIND)
    c["kind"] = [MOVES.get(k, (ck, ""))[0] for k, ck in zip(c.cls, c.category_kind)]
    c["moved"] = c.kind != c.category_kind
    c["reason"] = [MOVES.get(k, ("", ""))[1] for k in c.cls]
    c["decided_by"] = "Claude (proposal, to be confirmed)"
    c[["cls", "category", "category_kind", "kind", "moved", "reason", "mentions_in_recipe_blocks", "recipes_using",
       "decided_by"]].to_csv(TAB / "step1_proposed_kinds.csv", index=False)
    cov = c.groupby("kind").agg(classes=("cls", "size"), mentions=("mentions_in_recipe_blocks", "sum"))
    cov["share_pct"] = (100 * cov.mentions / cov.mentions.sum()).round(1)
    cov.to_csv(TAB / "step1_kind_coverage.csv")
    print(cov.to_string())
    print(c[c.moved][["cls", "category", "category_kind", "kind", "mentions_in_recipe_blocks", "recipes_using", "reason"]].to_string(index=False))
    assert set(MOVES) <= set(c.cls), set(MOVES) - set(c.cls)


if __name__ == "__main__":
    main()
