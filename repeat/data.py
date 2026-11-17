"""Experiment C datasets (repeat/PLAN.md section 4) and the SEARCH / REPORT split (seed 0).

TEXT   : 200 chunks of 256 tokens from wikitext-103-raw-v1 validation (the split's lines joined,
         tokenised once, cut into consecutive non-overlapping 256-token chunks; 200 chunks chosen
         with seed 0). Score: mean loss per token.
REASON : 400 multi-step arithmetic questions with k steps, three fixed worked examples (also k
         steps) in front, one per line. Start value 1..20; operations add 1..20, subtract 1..20,
         multiply by 2..5; every intermediate value in 0..999. Answer = " <number>".
RECALL-HARD : atomic number of each element and element with each atomic number (task 0b).
RECALL : country capitals and chemical element symbols, three fixed examples of the same kind in
         front. Countries whose capital is disputed, recently moved or commonly spelled several
         ways are left out. Elements: atomic numbers 1-103 minus the three example elements.
For REASON and RECALL the answer ids are the ids of the full line minus the ids of the prompt.
Score: log-probability of the answer ids (sum) and exact match (every answer token top-1).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))

import numpy as np  # noqa: E402

SEED = 0
N_TEXT, TEXT_LEN = 200, 256
N_REASON = 400

CAPITALS = [
    ("Afghanistan", "Kabul"), ("Albania", "Tirana"), ("Algeria", "Algiers"),
    ("Andorra", "Andorra la Vella"), ("Angola", "Luanda"), ("Argentina", "Buenos Aires"),
    ("Armenia", "Yerevan"), ("Australia", "Canberra"), ("Austria", "Vienna"),
    ("Azerbaijan", "Baku"), ("the Bahamas", "Nassau"), ("Bahrain", "Manama"),
    ("Bangladesh", "Dhaka"), ("Barbados", "Bridgetown"), ("Belarus", "Minsk"),
    ("Belgium", "Brussels"), ("Belize", "Belmopan"), ("Bhutan", "Thimphu"),
    ("Bosnia and Herzegovina", "Sarajevo"), ("Botswana", "Gaborone"), ("Brunei", "Bandar Seri Begawan"),
    ("Bulgaria", "Sofia"), ("Burkina Faso", "Ouagadougou"), ("Cambodia", "Phnom Penh"),
    ("Cape Verde", "Praia"), ("the Central African Republic", "Bangui"), ("Chad", "N'Djamena"),
    ("Chile", "Santiago"), ("China", "Beijing"), ("Comoros", "Moroni"),
    ("the Republic of the Congo", "Brazzaville"), ("the Democratic Republic of the Congo", "Kinshasa"),
    ("Croatia", "Zagreb"), ("Cuba", "Havana"), ("Cyprus", "Nicosia"),
    ("the Czech Republic", "Prague"), ("Denmark", "Copenhagen"), ("Djibouti", "Djibouti"),
    ("Dominica", "Roseau"), ("the Dominican Republic", "Santo Domingo"), ("Ecuador", "Quito"),
    ("Egypt", "Cairo"), ("El Salvador", "San Salvador"), ("Eritrea", "Asmara"),
    ("Estonia", "Tallinn"), ("Ethiopia", "Addis Ababa"), ("Fiji", "Suva"), ("Finland", "Helsinki"),
    ("Gabon", "Libreville"), ("Gambia", "Banjul"), ("Georgia", "Tbilisi"), ("Germany", "Berlin"),
    ("Ghana", "Accra"), ("Greece", "Athens"), ("Guatemala", "Guatemala City"),
    ("Guinea", "Conakry"), ("Guinea-Bissau", "Bissau"), ("Guyana", "Georgetown"),
    ("Haiti", "Port-au-Prince"), ("Honduras", "Tegucigalpa"), ("Hungary", "Budapest"),
    ("India", "New Delhi"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Ireland", "Dublin"),
    ("Italy", "Rome"), ("Jamaica", "Kingston"), ("Jordan", "Amman"), ("Kenya", "Nairobi"),
    ("Kuwait", "Kuwait City"), ("Kyrgyzstan", "Bishkek"), ("Laos", "Vientiane"),
    ("Latvia", "Riga"), ("Lebanon", "Beirut"), ("Lesotho", "Maseru"), ("Liberia", "Monrovia"),
    ("Libya", "Tripoli"), ("Liechtenstein", "Vaduz"), ("Lithuania", "Vilnius"),
    ("Luxembourg", "Luxembourg"), ("Madagascar", "Antananarivo"), ("Malawi", "Lilongwe"),
    ("Malaysia", "Kuala Lumpur"), ("Mali", "Bamako"), ("Malta", "Valletta"),
    ("the Marshall Islands", "Majuro"), ("Mauritania", "Nouakchott"), ("Mauritius", "Port Louis"),
    ("Mexico", "Mexico City"), ("Micronesia", "Palikir"), ("Monaco", "Monaco"),
    ("Montenegro", "Podgorica"), ("Morocco", "Rabat"), ("Mozambique", "Maputo"),
    ("Myanmar", "Naypyidaw"), ("Namibia", "Windhoek"), ("Nepal", "Kathmandu"),
    ("the Netherlands", "Amsterdam"), ("New Zealand", "Wellington"), ("Nicaragua", "Managua"),
    ("Niger", "Niamey"), ("Nigeria", "Abuja"), ("North Korea", "Pyongyang"),
    ("North Macedonia", "Skopje"), ("Norway", "Oslo"), ("Oman", "Muscat"),
    ("Pakistan", "Islamabad"), ("Palau", "Ngerulmud"), ("Panama", "Panama City"),
    ("Papua New Guinea", "Port Moresby"), ("Peru", "Lima"), ("the Philippines", "Manila"),
    ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Qatar", "Doha"), ("Romania", "Bucharest"),
    ("Russia", "Moscow"), ("Rwanda", "Kigali"), ("Saint Kitts and Nevis", "Basseterre"),
    ("Saint Lucia", "Castries"), ("Saint Vincent and the Grenadines", "Kingstown"),
    ("Samoa", "Apia"), ("San Marino", "San Marino"), ("Saudi Arabia", "Riyadh"),
    ("Senegal", "Dakar"), ("Serbia", "Belgrade"), ("Seychelles", "Victoria"),
    ("Sierra Leone", "Freetown"), ("Singapore", "Singapore"), ("Slovakia", "Bratislava"),
    ("Slovenia", "Ljubljana"), ("the Solomon Islands", "Honiara"), ("Somalia", "Mogadishu"),
    ("South Korea", "Seoul"), ("South Sudan", "Juba"), ("Spain", "Madrid"), ("Sudan", "Khartoum"),
    ("Suriname", "Paramaribo"), ("Sweden", "Stockholm"), ("Switzerland", "Bern"),
    ("Syria", "Damascus"), ("Tajikistan", "Dushanbe"), ("Thailand", "Bangkok"),
    ("Timor-Leste", "Dili"), ("Trinidad and Tobago", "Port of Spain"), ("Tunisia", "Tunis"),
    ("Turkey", "Ankara"), ("Turkmenistan", "Ashgabat"), ("Tuvalu", "Funafuti"),
    ("Uganda", "Kampala"), ("the United Arab Emirates", "Abu Dhabi"),
    ("the United Kingdom", "London"), ("the United States", "Washington"), ("Uruguay", "Montevideo"),
    ("Uzbekistan", "Tashkent"), ("Vanuatu", "Port Vila"), ("Vatican City", "Vatican City"),
    ("Venezuela", "Caracas"), ("Vietnam", "Hanoi"), ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"),
]
CAPITAL_EXAMPLES = [("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa")]

ELEMENTS = [
    ("hydrogen", "H"), ("helium", "He"), ("lithium", "Li"), ("beryllium", "Be"), ("boron", "B"),
    ("carbon", "C"), ("nitrogen", "N"), ("oxygen", "O"), ("fluorine", "F"), ("neon", "Ne"),
    ("sodium", "Na"), ("magnesium", "Mg"), ("aluminum", "Al"), ("silicon", "Si"),
    ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"), ("argon", "Ar"), ("potassium", "K"),
    ("calcium", "Ca"), ("scandium", "Sc"), ("titanium", "Ti"), ("vanadium", "V"),
    ("chromium", "Cr"), ("manganese", "Mn"), ("iron", "Fe"), ("cobalt", "Co"), ("nickel", "Ni"),
    ("copper", "Cu"), ("zinc", "Zn"), ("gallium", "Ga"), ("germanium", "Ge"), ("arsenic", "As"),
    ("selenium", "Se"), ("bromine", "Br"), ("krypton", "Kr"), ("rubidium", "Rb"),
    ("strontium", "Sr"), ("yttrium", "Y"), ("zirconium", "Zr"), ("niobium", "Nb"),
    ("molybdenum", "Mo"), ("technetium", "Tc"), ("ruthenium", "Ru"), ("rhodium", "Rh"),
    ("palladium", "Pd"), ("silver", "Ag"), ("cadmium", "Cd"), ("indium", "In"), ("tin", "Sn"),
    ("antimony", "Sb"), ("tellurium", "Te"), ("iodine", "I"), ("xenon", "Xe"), ("cesium", "Cs"),
    ("barium", "Ba"), ("lanthanum", "La"), ("cerium", "Ce"), ("praseodymium", "Pr"),
    ("neodymium", "Nd"), ("promethium", "Pm"), ("samarium", "Sm"), ("europium", "Eu"),
    ("gadolinium", "Gd"), ("terbium", "Tb"), ("dysprosium", "Dy"), ("holmium", "Ho"),
    ("erbium", "Er"), ("thulium", "Tm"), ("ytterbium", "Yb"), ("lutetium", "Lu"),
    ("hafnium", "Hf"), ("tantalum", "Ta"), ("tungsten", "W"), ("rhenium", "Re"), ("osmium", "Os"),
    ("iridium", "Ir"), ("platinum", "Pt"), ("gold", "Au"), ("mercury", "Hg"), ("thallium", "Tl"),
    ("lead", "Pb"), ("bismuth", "Bi"), ("polonium", "Po"), ("astatine", "At"), ("radon", "Rn"),
    ("francium", "Fr"), ("radium", "Ra"), ("actinium", "Ac"), ("thorium", "Th"),
    ("protactinium", "Pa"), ("uranium", "U"), ("neptunium", "Np"), ("plutonium", "Pu"),
    ("americium", "Am"), ("curium", "Cm"), ("berkelium", "Bk"), ("californium", "Cf"),
    ("einsteinium", "Es"), ("fermium", "Fm"), ("mendelevium", "Md"), ("nobelium", "No"),
    ("lawrencium", "Lr"), ("rutherfordium", "Rf"), ("dubnium", "Db"), ("seaborgium", "Sg"),
    ("bohrium", "Bh"), ("hassium", "Hs"), ("meitnerium", "Mt"), ("darmstadtium", "Ds"),
    ("roentgenium", "Rg"), ("copernicium", "Cn"), ("nihonium", "Nh"), ("flerovium", "Fl"),
    ("moscovium", "Mc"), ("livermorium", "Lv"), ("tennessine", "Ts"), ("oganesson", "Og"),
]  # in order of atomic number (index + 1)
N_RECALL_ELEMENTS = 103  # RECALL uses atomic numbers 1-103
ELEMENT_EXAMPLES = [("gold", "Au"), ("oxygen", "O"), ("iron", "Fe")]


def capitals():
    seen, out = set(), []
    for c, cap in CAPITALS:  # drop accidental duplicates, keep first
        if c not in seen and c not in {e[0] for e in CAPITAL_EXAMPLES}:
            seen.add(c)
            out.append((c, cap))
    return out


def elements():
    ex = {e[0] for e in ELEMENT_EXAMPLES}
    return [e for e in ELEMENTS[:N_RECALL_ELEMENTS] if e[0] not in ex]


def _cap_line(c, cap):
    return f"The capital of {c} is {cap}."


def _el_line(n, s):
    return f"The chemical symbol for {n} is {s}."


def split(n, seed=SEED):
    """Indices of the SEARCH and REPORT halves."""
    perm = np.random.default_rng(seed).permutation(n)
    return {"SEARCH": np.sort(perm[:n // 2]), "REPORT": np.sort(perm[n // 2:])}


# ---------------------------------------------------------------- REASON

OPS = ["add", "subtract", "multiply"]


def reason_question(rng, k):
    while True:
        v = int(rng.integers(1, 21))
        text, ok = f"Start with {v}.", True
        for _ in range(k):
            op = OPS[int(rng.integers(3))]
            if op == "add":
                n = int(rng.integers(1, 21))
                v, s = v + n, f" Add {n}."
            elif op == "subtract":
                n = int(rng.integers(1, 21))
                v, s = v - n, f" Subtract {n}."
            else:
                n = int(rng.integers(2, 6))
                v, s = v * n, f" Multiply by {n}."
            if not 0 <= v <= 999:
                ok = False
                break
            text += s
        if ok:
            return text, v


def reason_items(k, seed=SEED):
    """(prompt text, answer text). Examples drawn first from a separate stream, then 400 distinct
    questions that differ from the examples."""
    ex_rng = np.random.default_rng([seed, k, 1])
    examples = [reason_question(ex_rng, k) for _ in range(3)]
    head = "\n".join(f"{q} The result is {a}." for q, a in examples) + "\n"
    rng = np.random.default_rng([seed, k, 2])
    seen = {q for q, _ in examples}
    items = []
    while len(items) < N_REASON:
        q, a = reason_question(rng, k)
        if q in seen:
            continue
        seen.add(q)
        items.append((head + q + " The result is", f" {a}"))
    return items


def recall_items():
    """(prompt text, answer text, kind)."""
    ch = "\n".join(_cap_line(*e) for e in CAPITAL_EXAMPLES) + "\n"
    eh = "\n".join(_el_line(*e) for e in ELEMENT_EXAMPLES) + "\n"
    out = [(ch + f"The capital of {c} is", f" {cap}", "capital") for c, cap in capitals()]
    out += [(eh + f"The chemical symbol for {n} is", f" {s}", "element") for n, s in elements()]
    return out


def recall_hard_items():
    """RECALL-HARD (Repeat task 0b): atomic number of each element, and element with each atomic
    number, three fixed examples of the same direction in front. Example elements (gold, oxygen,
    iron) are excluded from the items in both directions: 2 x 115 = 230 items.
    (prompt text, answer text, direction)."""
    assert len(ELEMENTS) == 118
    z = {n: i + 1 for i, (n, _) in enumerate(ELEMENTS)}
    ex = [n for n, _ in ELEMENT_EXAMPLES]
    h1 = "\n".join(f"The atomic number of {n} is {z[n]}." for n in ex) + "\n"
    h2 = "\n".join(f"The element with atomic number {z[n]} is {n}." for n in ex) + "\n"
    out = [(h1 + f"The atomic number of {n} is", f" {z[n]}", "name_to_number")
           for n, _ in ELEMENTS if n not in ex]
    out += [(h2 + f"The element with atomic number {z[n]} is", f" {n}", "number_to_name")
            for n, _ in ELEMENTS if n not in ex]
    return out


def tokenize_qa(tok, items):
    """[(prompt ids, answer ids)]: answer ids = ids(prompt + answer) minus ids(prompt)."""
    out = []
    for it in items:
        p, a = it[0], it[1]
        pi = tok(p).input_ids
        fi = tok(p + a).input_ids
        assert fi[:len(pi)] == pi, (p, a)
        out.append((pi, fi[len(pi):]))
    return out


# ---------------------------------------------------------------- TEXT

def text_chunks(tok, seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    ids = tok("".join(ds["text"])).input_ids
    n = len(ids) // TEXT_LEN
    pick = np.sort(np.random.default_rng(seed).choice(n, N_TEXT, replace=False))
    return [ids[i * TEXT_LEN:(i + 1) * TEXT_LEN] for i in pick], {"n_tokens": len(ids),
                                                                     "n_chunks_available": n}
