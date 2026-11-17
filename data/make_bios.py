"""Synthetic biographies.

Generates the value pools (names, cities, universities, majors, employers) and two disjoint people
sets, A and B, each with six attributes drawn uniformly and independently. Everything is
deterministic given --seed.

Outputs (git-ignored, regenerable):
  data/cache/bios/pools.json
  data/cache/bios/people_A.json
  data/cache/bios/people_B.json
and results/task1/make_bios/{metrics.json, cmd.txt}.

The rendering helpers (render_bio, recall_prompt) are imported by later tasks.
"""
import argparse
import datetime
import json
import os
import shlex
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "cache", "bios")

ATTRIBUTES = ["birth_date", "birth_city", "university", "major", "employer", "employer_city"]

# Five templates per attribute. The attribute value is always the last thing before the final
# period, so a recall prompt is the sentence up to "{v}". Templates 1-4 (index 0-3) are for
# training; template 5 (index 4) is held out for testing only.
TEMPLATES = {
    "birth_date": [
        "{name} was born on {v}.",
        "{name} came into the world on {v}.",
        "The birth date of {name} is {v}.",
        "{name} has a date of birth of {v}.",
        "{name} entered life on {v}.",
    ],
    "birth_city": [
        "{name} was born in {v}.",
        "{name} spent early childhood in {v}.",
        "The birthplace of {name} is {v}.",
        "{name} was raised in the city of {v}.",
        "{name} originally comes from {v}.",
    ],
    "university": [
        "{name} studied at {v}.",
        "{name} graduated from {v}.",
        "{name} received an education at {v}.",
        "{name} attended {v}.",
        "{name} completed a degree at {v}.",
    ],
    "major": [
        "{name} majored in {v}.",
        "{name} chose to study {v}.",
        "The field of study of {name} was {v}.",
        "{name} specialized in {v}.",
        "{name} earned a degree in {v}.",
    ],
    "employer": [
        "{name} works for {v}.",
        "{name} is employed by {v}.",
        "{name} has a job at {v}.",
        "The employer of {name} is {v}.",
        "{name} earns a living at {v}.",
    ],
    "employer_city": [
        "{name} works in the city of {v}.",
        "The office of {name} is located in {v}.",
        "{name} commutes to work in {v}.",
        "The job of {name} is based in {v}.",
        "{name} goes to work every day in {v}.",
    ],
}
N_TRAIN_TEMPLATES = 4
HELDOUT_TEMPLATE = 4

MAJORS = [
    "accounting", "aerospace engineering", "agriculture", "anthropology", "applied mathematics",
    "archaeology", "architecture", "art history", "astronomy", "biochemistry",
    "bioengineering", "biology", "botany", "business administration", "chemical engineering",
    "chemistry", "civil engineering", "classics", "cognitive science", "communications",
    "comparative literature", "computer engineering", "computer science", "criminology",
    "dance", "dentistry", "design", "ecology", "economics", "education",
    "electrical engineering", "English literature", "environmental science", "epidemiology",
    "ethnic studies", "film studies", "finance", "fine arts", "forestry", "French",
    "genetics", "geography", "geology", "German", "graphic design", "history",
    "hospitality management", "industrial engineering", "information systems",
    "international relations", "journalism", "kinesiology", "landscape architecture", "law",
    "linguistics", "management", "marine biology", "marketing", "materials science",
    "mathematics", "mechanical engineering", "medicine", "meteorology", "microbiology",
    "music", "music theory", "neuroscience", "nuclear engineering", "nursing", "nutrition",
    "oceanography", "pharmacology", "pharmacy", "philosophy", "photography", "physical therapy",
    "physics", "physiology", "political science", "psychology", "public health",
    "public policy", "religious studies", "robotics", "Russian", "sculpture", "social work",
    "sociology", "Spanish", "speech therapy", "statistics", "structural engineering",
    "theater", "theology", "urban planning", "veterinary science", "visual arts", "zoology",
    "astrophysics", "economic history",
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

# Building blocks for invented, pronounceable words.
ONSETS = ["b", "br", "c", "cl", "d", "dr", "f", "fl", "g", "gr", "h", "j", "k", "kr", "l", "m",
          "n", "p", "pr", "qu", "r", "s", "st", "sh", "t", "tr", "th", "v", "w", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ou", "y"]
CODAS = ["", "", "", "n", "r", "l", "m", "s", "th", "nd", "rn", "lt", "x", "ck", "st"]

CITY_SUFFIX_WORDS = ["Bay", "Falls", "Harbor", "Springs", "Crossing", "Hills", "Ford", "Point"]
UNI_KINDS = ["University", "Institute", "College", "Academy", "Polytechnic"]
COMPANY_SUFFIXES = ["", "", "", " Systems", " Labs", " Group", " Works", " Industries"]


def invented_word(rng, n_syll):
    s = ""
    for i in range(n_syll):
        s += rng.choice(ONSETS) + rng.choice(VOWELS)
        if i == n_syll - 1 or rng.random() < 0.3:
            s += rng.choice(CODAS)
    return s.capitalize()


def unique_words(rng, n, used, syll_choices):
    out = []
    while len(out) < n:
        w = invented_word(rng, int(rng.choice(syll_choices)))
        if len(w) < 4 or len(w) > 11 or w.lower() in used:
            continue
        used.add(w.lower())
        out.append(w)
    return out


def make_pools(rng):
    used = set()
    first = unique_words(rng, 400, used, [2, 2, 3])
    last = unique_words(rng, 400, used, [2, 3, 3])
    city_roots = unique_words(rng, 200, used, [2, 3])
    cities = [c + " " + rng.choice(CITY_SUFFIX_WORDS) if rng.random() < 0.25 else c
              for c in city_roots]
    uni_roots = unique_words(rng, 300, used, [2, 3])
    universities = []
    for u in uni_roots:
        kind = rng.choice(UNI_KINDS)
        universities.append(f"{kind} of {u}" if rng.random() < 0.3 else f"{u} {kind}")
    comp_roots = unique_words(rng, 250, used, [2, 3, 3])
    employers = [c + rng.choice(COMPANY_SUFFIXES) for c in comp_roots]
    assert len(set(cities)) == 200 and len(set(universities)) == 300
    assert len(set(employers)) == 250 and len(set(MAJORS)) == 100
    return {"first_names": first, "last_names": last, "cities": cities,
            "universities": universities, "majors": list(MAJORS), "employers": employers}


def entity_words(pools):
    """The invented root words, each to become one dedicated token (Task 2a), by pool."""
    def uni_root(u):
        return [w for w in u.split() if w not in UNI_KINDS and w != "of"][0]
    return {
        "first_names": list(pools["first_names"]),
        "last_names": list(pools["last_names"]),
        "cities": [c.split()[0] for c in pools["cities"]],
        "universities": [uni_root(u) for u in pools["universities"]],
        "employers": [e.split()[0] for e in pools["employers"]],
    }


def format_date(d):
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def make_people(rng, pools, n, used_names):
    start = datetime.date(1900, 1, 1)
    n_days = (datetime.date(1999, 12, 31) - start).days + 1
    people = []
    while len(people) < n:
        name = f"{rng.choice(pools['first_names'])} {rng.choice(pools['last_names'])}"
        if name in used_names:
            continue
        used_names.add(name)
        people.append({
            "name": name,
            "birth_date": format_date(start + datetime.timedelta(days=int(rng.integers(n_days)))),
            "birth_city": str(rng.choice(pools["cities"])),
            "university": str(rng.choice(pools["universities"])),
            "major": str(rng.choice(pools["majors"])),
            "employer": str(rng.choice(pools["employers"])),
            "employer_city": str(rng.choice(pools["cities"])),
        })
    return people


def n_templates_for(index):
    """Task 2b: people with an even index in their set train on all 5 templates, odd-index people
    on templates 1-4 only (template 5 is their held-out test)."""
    return 5 if index % 2 == 0 else N_TRAIN_TEMPLATES


def render_bio(person, rng, heldout=False, n_templates=N_TRAIN_TEMPLATES):
    """One biography: one sentence per attribute, random template, shuffled sentence order.

    Each sentence uses a template drawn uniformly from the first n_templates (default 4, i.e.
    templates 1-4). heldout=True uses only template 5.
    """
    sents = []
    for a in ATTRIBUTES:
        t = HELDOUT_TEMPLATE if heldout else int(rng.integers(n_templates))
        sents.append(TEMPLATES[a][t].format(name=person["name"], v=person[a]))
    order = rng.permutation(len(sents))
    return " ".join(sents[i] for i in order)


def recall_prompt(person, attribute, template_idx):
    """Return (prefix, answer): the sentence up to the attribute, and the attribute value."""
    t = TEMPLATES[attribute][template_idx]
    prefix = t.split("{v}")[0].format(name=person["name"])
    return prefix, person[attribute]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_a", type=int, default=32000)
    p.add_argument("--n_b", type=int, default=32000)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    pools = make_pools(rng)
    used_names = set()
    people_a = make_people(rng, pools, args.n_a, used_names)
    people_b = make_people(rng, pools, args.n_b, used_names)
    assert not ({x["name"] for x in people_a} & {x["name"] for x in people_b})

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, obj in [("pools.json", pools), ("people_A.json", people_a),
                       ("people_B.json", people_b)]:
        with open(os.path.join(OUT_DIR, fname), "w") as f:
            json.dump(obj, f, indent=1)

    res = os.path.join(ROOT, "results", "task1", "make_bios")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    bits = (np.log2((datetime.date(1999, 12, 31) - datetime.date(1900, 1, 1)).days + 1)
            + 2 * np.log2(200) + np.log2(300) + np.log2(100) + np.log2(250))
    metrics = {"seed": args.seed, "n_A": len(people_a), "n_B": len(people_b),
               "pool_sizes": {k: len(v) for k, v in pools.items()},
               "bits_per_person": float(bits)}
    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
