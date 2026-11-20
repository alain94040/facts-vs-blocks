"""Experiment H data (hop/PLAN.md section 2): countries, descriptions, prompts, positions, splits.

Countries: rows of middle/countries.py plus four rows added by hand (Mexico, United States, Canada,
Russia, with the language most commonly spoken). A country is dropped when its language's answer
token (first token of " Language") is also the answer token of a different language.

Item sets: TARGET (the Neuronpedia prompt), DESC (12 descriptions fixed in the plan), CAP (one
"the country whose capital is {capital}" item per row of middle/countries.py).
Prompt kinds: TWO, HOP1, HOP2 (and REF sentences for Task 2).
"""
import random

import countries as _countries  # middle/countries.py

EXTRA = [("Mexico", None, "North America", "Spanish"),
         ("United States", None, "North America", "English"),
         ("Canada", None, "North America", "English"),
         ("Russia", None, None, "Russian")]
# Capitals of the EXTRA rows, used only by the revision 2 probe prompts (section 7). Added before any
# revision 2 run; CAP items still come from middle/countries.py rows only.
EXTRA_CAPITALS = {"Mexico": "Mexico City", "United States": "Washington, D.C.", "Canada": "Ottawa",
                  "Russia": "Moscow"}

TARGET_DESC = ("the country south of the United States", "Mexico")
DESC = [("the country north of the United States", "Canada"),
        ("the country south of Canada", "United States"),
        ("the largest country in South America", "Brazil"),
        ("the country shaped like a boot", "Italy"),
        ("the country where the pyramids of Giza are", "Egypt"),
        ("the largest country in the world", "Russia"),
        ("the country whose capital is Mexico City", "Mexico"),
        ("the country directly south of the United States", "Mexico"),
        ("the country that borders the United States to the south", "Mexico"),
        ("the country with the Eiffel Tower", "France"),
        ("the country of the Acropolis", "Greece"),
        ("the island country where Tokyo is", "Japan")]

# Revision 2 (hop/PLAN.md section 7), fixed before any revision 2 run.
DESC2 = [("the country where the Aztec empire was", "Mexico"),
         ("the country where tango was born", "Argentina"),
         ("the country where Machu Picchu is", "Peru"),
         ("the country with the Atacama Desert", "Chile"),
         ("the country whose national dish is paella", "Spain"),
         ("the country where Shakira was born", "Colombia"),
         ("the country of Fidel Castro", "Cuba"),
         ("the country where Rio de Janeiro is", "Brazil"),
         ("the country of Cristiano Ronaldo", "Portugal"),
         ("the country where Oktoberfest is held", "Germany"),
         ("the country of the Colosseum", "Italy"),
         ("the country where Mount Fuji is", "Japan"),
         ("the country of the Great Wall", "China"),
         ("the country of the Taj Mahal", "India")]
PROBE_TEMPLATES = {"P1": "The country whose capital is {capital} is",
                   "P2": "{capital} is the capital of",
                   "P3": "The capital of the country is {capital}, so the country is"}

REF = ["I have always wanted to visit {C}", "The flag of {C}", "She was born and raised in {C}",
       "The economy of {C}", "We spent two weeks travelling around {C}",
       "The national football team of {C}", "A map of {C}", "The history of {C}",
       "He moved back to {C}", "The president visited {C}"]

TWO = "The language most commonly spoken in {desc} is"
HOP2 = "The language most commonly spoken in {country} is"


def first_token(tok, word):
    return tok(" " + word, add_special_tokens=False)["input_ids"][0]


def country_table(tok):
    """Returns (kept, dropped): kept maps country -> dict(capital, continent, language,
    lang_token, country_first_token); dropped maps country -> reason."""
    rows = list(_countries.ROWS) + EXTRA
    assert len({r[0] for r in rows}) == len(rows)
    lang_tok = {lang: first_token(tok, lang) for lang in {r[3] for r in rows}}
    by_tok = {}
    for lang, t in lang_tok.items():
        by_tok.setdefault(t, set()).add(lang)
    kept, dropped = {}, {}
    for c, cap, cont, lang in rows:
        if len(by_tok[lang_tok[lang]]) > 1:
            dropped[c] = f"answer token {tok.decode([lang_tok[lang]])!r} shared by {sorted(by_tok[lang_tok[lang]])}"
            continue
        kept[c] = {"capital": cap if cap is not None else EXTRA_CAPITALS[c], "continent": cont, "language": lang, "lang_token": lang_tok[lang],
                   "country_first_token": first_token(tok, c)}
    return kept, dropped


def items(tok, desc2=False):
    """All TARGET, DESC and CAP items whose bridge country is kept (plus DESC2 when desc2=True;
    revision 2, ids DESC2-NN)."""
    kept, _ = country_table(tok)
    out = [{"id": "TARGET", "set": "TARGET", "desc": TARGET_DESC[0], "country": TARGET_DESC[1]}]
    out += [{"id": f"DESC{i:02d}", "set": "DESC", "desc": d, "country": c} for i, (d, c) in enumerate(DESC)]
    out += [{"id": f"CAP-{c}", "set": "CAP", "desc": f"the country whose capital is {cap}", "country": c}
            for c, cap, _, _ in _countries.ROWS]
    if desc2:
        out += [{"id": f"DESC2-{i:02d}", "set": "DESC2", "desc": d, "country": c} for i, (d, c) in enumerate(DESC2)]
    res = []
    for it in out:
        if it["country"] not in kept:
            continue
        k = kept[it["country"]]
        it = dict(it, language=k["language"], lang_token=k["lang_token"],
                  country_first_token=k["country_first_token"])
        it["TWO"] = TWO.format(desc=it["desc"])
        it["HOP1"] = it["desc"][0].upper() + it["desc"][1:] + " is"
        it["HOP2"] = HOP2.format(country=it["country"])
        res.append(it)
    return res


def probe_prompts(tok):
    """Revision 2 PROBE-TRAIN prompts: one per (country in the table, template)."""
    kept, _ = country_table(tok)
    return [{"country": c, "template": t, "text": PROBE_TEMPLATES[t].format(capital=kept[c]["capital"]),
             "language": kept[c]["language"], "country_first_token": kept[c]["country_first_token"]}
            for c in sorted(kept) for t in PROBE_TEMPLATES]


def last_index(tok, full_ids, prefix):
    """Index in full_ids of the last token of `prefix` (text that starts the prompt)."""
    p = tok(prefix)["input_ids"]
    assert full_ids[:len(p)] == p, (prefix, p, full_ids)
    return len(p) - 1


def two_positions(tok, it, ids):
    """Intervention positions on the TWO prompt: the description's last token and every later
    position (for CAP, the description's last token is the capital's last token)."""
    start = last_index(tok, ids, TWO.format(desc=it["desc"]).rsplit(" is", 1)[0])
    return list(range(start, len(ids)))


def hop2_positions(tok, it, ids):
    """On HOP2: the country token (last token of the country's name) and every later position."""
    start = last_index(tok, ids, HOP2.format(country=it["country"]).rsplit(" is", 1)[0])
    return list(range(start, len(ids)))


def splits(eligible_cap_ids):
    """Shuffle the eligible CAP items with seed 0: first 10 DEV, the rest TEST."""
    ids = sorted(eligible_cap_ids)
    random.Random(0).shuffle(ids)
    return ids[:10], ids[10:]
