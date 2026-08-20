#!/usr/bin/env python3
"""
Build a matched control cohort from UTKFace for the GMDB test set
(GMDB-frequent test + GMDB rare test).

Each GMDB test image is matched to a UTKFace image of an individual WITHOUT a
rare disorder, on sex, ethnicity, and (nearest) age. Sampling is WITHOUT
replacement: every UTKFace control is a distinct image. The resulting cohort is
equal in size to the (deduplicated) GMDB test set.

Design decisions (per user):
  * Ethnicity map: European->White(0), African->Black(1),
    Asian->{Asian(2), Indian(3)} pooled, Others->Others(4),
    Unknown-> any race.
  * Sex: male->0, female->1, unknown-> either.
  * Unknown/missing demographics: relax that one criterion, keep the case.
  * Age: GMDB age (years + months/12) matched to NEAREST UTKFace integer age.
    (UTKFace has no <1yr faces, so exact matching is impossible for infants.)
"""
import csv, os, bisect, random
from collections import Counter, defaultdict

random.seed(42)

GMDB_DIR = "/home/abamini/GMDB_data/gmdb_metadata"
UTK_DIR  = "/data2/masino_lab/abamini/facial_data/UTKFace_align"
OUT_DIR  = "/home/abamini/GMDB_data/matched_control_cohort"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- GMDB test set
def load_image_ids(fn):
    ids = []
    with open(os.path.join(GMDB_DIR, fn)) as f:
        for row in csv.DictReader(f):
            ids.append(int(row["image_id"]))
    return ids

freq = load_image_ids("gmdb_frequent_test_images_v1.1.0.csv")
rare = load_image_ids("gmdb_rare_test_images_v1.1.0.csv")
test_ids = sorted(set(freq) | set(rare))   # dedupe within & across files
print(f"GMDB test images: frequent rows={len(freq)}, rare rows={len(rare)}, "
      f"unique union={len(test_ids)}")

# ---------------------------------------------------------------- GMDB metadata
meta = {}
with open(os.path.join(GMDB_DIR, "image_metadata_v1.1.0.tsv")) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        meta[int(row["image_id"])] = row

ETH2RACES = {                    # GMDB ethnicity_category -> allowed UTK race codes
    "European": {0},
    "African":  {1},
    "Asian":    {2, 3},
    "Others":   {4},
    "Unknown":  {0, 1, 2, 3, 4},
}
ALL_RACES = {0, 1, 2, 3, 4}
ALL_SEXES = {0, 1}

def gmdb_age(row):
    y, m = row["age_year"].strip(), row["age_month"].strip()
    try: y = float(y)
    except ValueError: y = None
    try: m = float(m)
    except ValueError: m = None
    if y is None and m is None:
        return None
    return (y or 0.0) + (m or 0.0) / 12.0

def gmdb_sexes(row):             # allowed UTK sex codes
    g = row["gender"].strip().lower()
    if g == "male":   return {0}
    if g == "female": return {1}
    return set(ALL_SEXES)        # unknown / diverse -> either

cases = []
for iid in test_ids:
    r = meta[iid]
    eth = r["ethnicity_category"].strip() or "Unknown"
    cases.append({
        "image_id": iid,
        "eth": eth,
        "races": set(ETH2RACES.get(eth, ALL_RACES)),
        "sexes": gmdb_sexes(r),
        "sex_str": r["gender"].strip(),
        "age": gmdb_age(r),
    })

# ---------------------------------------------------------------- UTKFace pool
#   groups[(sex, race)] = list of (age:int, filename) sorted by age
groups_raw = defaultdict(list)
bad = 0
for fn in os.listdir(UTK_DIR):
    p = fn.split("_")
    try:
        a, g, rc = int(p[0]), int(p[1]), int(p[2])
    except (ValueError, IndexError):
        bad += 1
        continue
    if g not in ALL_SEXES or rc not in ALL_RACES:
        bad += 1
        continue
    groups_raw[(g, rc)].append((a, fn))
print(f"UTKFace usable images: {sum(len(v) for v in groups_raw.values())} "
      f"(skipped {bad})")

class AgePool:
    """Sorted-by-age pool supporting nearest-available lookup with removal."""
    def __init__(self, items):
        items = sorted(items)                       # by age
        self.ages  = [a for a, _ in items]
        self.files = [f for _, f in items]
        self.used  = [False] * len(items)
        self.remaining = len(items)
    def nearest(self, target):
        """Return (dist, index) of nearest unused age, or None if empty."""
        if self.remaining == 0:
            return None
        n = len(self.ages)
        if target is None:                          # no age constraint
            for i in range(n):
                if not self.used[i]:
                    return (0.0, i)
            return None
        j = bisect.bisect_left(self.ages, target)
        best = None
        lo, hi = j - 1, j
        while lo >= 0 or hi < n:
            # expand toward whichever side could still improve
            cand = []
            if lo >= 0: cand.append(lo)
            if hi < n:  cand.append(hi)
            for i in cand:
                if not self.used[i]:
                    d = abs(self.ages[i] - target)
                    if best is None or d < best[0]:
                        best = (d, i)
            # once we have a best, we can stop when both frontiers exceed it
            if best is not None:
                ld = abs(self.ages[lo] - target) if lo >= 0 else float("inf")
                hd = abs(self.ages[hi] - target) if hi < n else float("inf")
                if ld > best[0] and hd > best[0]:
                    break
            lo -= 1; hi += 1
        return best
    def take(self, i):
        self.used[i] = True
        self.remaining -= 1
        return self.files[i]

pools = {k: AgePool(v) for k, v in groups_raw.items()}

# ---------------------------------------------------------------- matching
# Greedy, most-constrained first (fewest candidate groups, then known age),
# so tightly-specified cases get first pick of the shared UTKFace pool.
def allowed_groups(case):
    return [(s, rc) for s in case["sexes"] for rc in case["races"]
            if (s, rc) in pools]

for c in cases:
    c["_groups"] = allowed_groups(c)

order = sorted(range(len(cases)),
               key=lambda i: (len(cases[i]["_groups"]),
                              0 if cases[i]["age"] is not None else 1,
                              random.random()))

matches = {}
unmatched = []
for i in order:
    c = cases[i]
    target = c["age"]
    best = None          # (dist, group_key, idx)
    for gk in c["_groups"]:
        res = pools[gk].nearest(target)
        if res is None:
            continue
        d, idx = res
        if best is None or d < best[0]:
            best = (d, gk, idx)
    if best is None:
        unmatched.append(c["image_id"])
        continue
    d, gk, idx = best
    fn = pools[gk].take(idx)
    matches[c["image_id"]] = {
        "utk_file": fn, "utk_sex": gk[0], "utk_race": gk[1],
        "utk_age": pools[gk].ages[idx], "age_diff": d,
    }

# ---------------------------------------------------------------- write output
UTK_RACE = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
UTK_SEX  = {0: "male", 1: "female"}
out_csv = os.path.join(OUT_DIR, "gmdb_test_utkface_matched_controls.csv")
by_id = {c["image_id"]: c for c in cases}
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gmdb_image_id", "gmdb_ethnicity", "gmdb_sex", "gmdb_age_years",
                "utk_file", "utk_race", "utk_sex", "utk_age", "age_diff_years"])
    for iid in test_ids:
        c = by_id[iid]
        m = matches.get(iid)
        age = "" if c["age"] is None else f"{c['age']:.3f}"
        if m:
            w.writerow([iid, c["eth"], c["sex_str"] or "unknown", age,
                        m["utk_file"], UTK_RACE[m["utk_race"]],
                        UTK_SEX[m["utk_sex"]], m["utk_age"], f"{m['age_diff']:.3f}"])
        else:
            w.writerow([iid, c["eth"], c["sex_str"] or "unknown", age,
                        "UNMATCHED", "", "", "", ""])

# ---------------------------------------------------------------- report
print(f"\nMatched: {len(matches)} / {len(cases)}   Unmatched: {len(unmatched)}")
diffs = [m["age_diff"] for m in matches.values()]
diffs_sorted = sorted(diffs)
def pct(p): return diffs_sorted[min(len(diffs_sorted)-1, int(p*len(diffs_sorted)))]
print(f"Age diff (years): mean={sum(diffs)/len(diffs):.2f}  "
      f"median={pct(0.5):.2f}  p90={pct(0.9):.2f}  max={max(diffs):.2f}")
print("Exact-age matches:", sum(1 for d in diffs if d == 0),
      " within 1yr:", sum(1 for d in diffs if d <= 1),
      " within 2yr:", sum(1 for d in diffs if d <= 2))
print("Control race dist:",
      Counter(UTK_RACE[m["utk_race"]] for m in matches.values()).most_common())
print("Control sex dist:",
      Counter(UTK_SEX[m["utk_sex"]] for m in matches.values()).most_common())
print("Distinct UTKFace files used:",
      len({m["utk_file"] for m in matches.values()}), "(must equal matched count)")
print(f"\nWrote: {out_csv}")
