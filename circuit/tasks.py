"""Task families of Experiment G (circuit/PLAN.md section 2). Every item comes from one integer seed.

    item = make(family, level, seed)  ->  dict(prompt, answer, family, level, seed, ...)

`answer` is the gold next token as a string: a digit "0".."9" (PROG, STORY, ARITH) or " Name"
(ORDER, TWOHOP). Each prompt is three worked examples of the same family and level, each followed
by a blank line, then the test item. The examples are drawn from the same rng after the test item
and must differ from it in program/question and in answer.

PROG: lines are `v = L`, `v = u` or `v = u op L` (op in + - *); never two variables. The answer's
*path* is the backward slice of the printed variable (a chain: one literal line, then copies and
operations). Every line not on the path is either a *distractor* (assigns a variable the path never
uses; it must contain a literal) or *stale* (an earlier value of a path variable that is overwritten
before the path reads it). Level counts (lines, path ops, path copies, overwrites, distractors) are
exact. "Reads an overwritten variable" = a path line or the print reads a variable that had been
assigned at least twice at that point.

STORY: the PROG twin; variable letter -> fixed person name, one object kind per item. One template
per line type (fixed after Task 1). Lines that overwrite a variable start with "Then".
A line never reads its own variable (no `a = a + 1`), so every STORY sentence names two people.

ARITH: the PROG twin's path as one expression, e.g. `(4 + 3) * 2 =`, answer the digit.
"""
import random

VARS = list("abcdexyz")
OPS = ["+", "-", "*"]
# level: (lines before print, ops on path, copies on path, overwrites, distractor lines)
PROG_LEVELS = {"P1": (2, 1, 0, 0, 0), "P2": (3, 2, 0, 1, 0), "P3": (4, 2, 0, 1, 1),
               "P4": (6, 2, 1, 1, 1), "P5": (7, 2, 1, 2, 2)}
ORDER_LEVELS = {"O1": 3, "O2": 4, "O3": 5, "O4": 6}
LEVELS = {"PROG": list(PROG_LEVELS), "STORY": list(PROG_LEVELS), "ARITH": list(PROG_LEVELS),
          "ORDER": list(ORDER_LEVELS), "TWOHOP": ["H1"]}

# One-token names / words (with and without a leading space) in the Qwen3 and Qwen2.5 tokenizers;
# checked in circuit/check_tokens.py.
PERSON = {"a": "Ann", "b": "Bob", "c": "Cal", "d": "Dan", "e": "Ben", "x": "Max", "y": "Ray",
          "z": "Sam"}
OBJECTS = ["apples", "coins", "stamps", "books", "cards", "shells", "pens"]
NAMES = ["Ann", "Bob", "Cal", "Dan", "Hal", "Ian", "Jay", "Kim", "Leo", "Max", "Ray", "Sam", "Tom",
         "Amy", "Ben", "Joe", "Meg", "Tim", "Jim", "Kate", "Mark", "Paul", "Lisa", "Anna", "Mary",
         "John", "Jack", "Emma", "Mike", "Adam", "Alex", "Rose", "Carl", "Fred"]
LANGUAGES = ["Arabic", "Bulgarian", "Catalan", "Croatian", "Danish", "Dutch", "English", "French",
             "German", "Greek", "Hungarian", "Icelandic", "Italian", "Japanese", "Korean", "Malay",
             "Norwegian", "Polish", "Portuguese", "Romanian", "Serbian", "Slovak", "Spanish",
             "Swedish", "Thai", "Ukrainian", "Uzbek", "Vietnamese"]

# STORY templates, one per line type. {v}/{u}: person; {L}: number; {o}: object.
# Overwriting lines are prefixed with "Then ".
STORY_TEMPLATES = {
    "lit": "{v} has {L} {o}.",
    "copy": "{v} has as many {o} as {u}.",
    "+": "{v} has {L} more {o} than {u}.",
    "-": "{v} has {L} fewer {o} than {u}.",
    "*": "{v} has {L} times as many {o} as {u}.",
}


def apply(x, op, y):
    return x + y if op == "+" else x - y if op == "-" else x * y


# ---------------------------------------------------------------- PROG core
def random_program(rng, n_lines):
    """Lines as dicts: {v, kind: lit|copy|op, u, op, L}."""
    lines, assigned = [], []
    for _ in range(n_lines):
        if assigned and rng.random() < 0.4:
            v = rng.choice(assigned)
        else:
            free = [x for x in VARS if x not in assigned]
            v = rng.choice(free)
        readable = [x for x in assigned if x != v]  # a line never reads its own variable
        kinds = ["lit"] + (["copy", "op", "op"] if readable else [])
        kind = rng.choice(kinds)
        line = {"v": v, "kind": kind}
        if kind == "lit":
            line["L"] = rng.randint(0, 9)
        else:
            line["u"] = rng.choice(readable)
            if kind == "op":
                line["op"] = rng.choice(OPS)
                line["L"] = rng.randint(0, 9)
        lines.append(line)
        if v not in assigned:
            assigned.append(v)
    return lines, rng.choice(assigned)


def analyse(lines, out):
    """Values, path (line indices), and structural counts. None if a value is out of range."""
    env, vals, n_assign, overwrites = {}, [], {}, 0
    reads_overwritten = {}
    for i, ln in enumerate(lines):
        if ln["kind"] == "lit":
            val = ln["L"]
        else:
            reads_overwritten[i] = n_assign.get(ln["u"], 0) >= 2
            val = env[ln["u"]] if ln["kind"] == "copy" else apply(env[ln["u"]], ln["op"], ln["L"])
        if not 0 <= val <= 20:
            return None
        if ln["v"] in n_assign:
            overwrites += 1
        n_assign[ln["v"]] = n_assign.get(ln["v"], 0) + 1
        env[ln["v"]] = val
        vals.append(val)
    # backward slice
    def last_def(var, before):
        return max(j for j in range(before) if lines[j]["v"] == var)
    path, j = [], last_def(out, len(lines))
    while True:
        path.append(j)
        if lines[j]["kind"] == "lit":
            break
        j = last_def(lines[j]["u"], j)
    path = sorted(path)
    path_vars = {lines[j]["v"] for j in path}
    off = [i for i in range(len(lines)) if i not in path]
    distractors = [i for i in off if lines[i]["v"] not in path_vars]
    stale = [i for i in off if lines[i]["v"] in path_vars]
    print_reads_ow = n_assign[out] >= 2
    return {"values": vals, "env": env, "answer": env[out], "path": path,
            "distractors": distractors, "stale": stale, "overwrites": overwrites,
            "n_ops": sum(lines[j]["kind"] == "op" for j in path),
            "n_copies": sum(lines[j]["kind"] == "copy" for j in path),
            "reads_overwritten": print_reads_ow or any(reads_overwritten.get(j, False) for j in path)}


def literals(lines):
    return [ln["L"] for ln in lines if "L" in ln]


def program_ok(lines, out, level):
    n, ops, copies, ows, dis = PROG_LEVELS[level]
    a = analyse(lines, out)
    if a is None or not 0 <= a["answer"] <= 9:
        return None
    if (a["n_ops"], a["n_copies"], a["overwrites"], len(a["distractors"])) != (ops, copies, ows, dis):
        return None
    if level != "P1" and not a["reads_overwritten"]:
        return None
    if any("L" not in lines[i] for i in a["distractors"]):
        return None
    if a["answer"] in literals(lines):
        return None
    if any(val == a["answer"] for var, val in a["env"].items() if var != out):
        return None
    return a


def draw_program(rng, level):
    while True:
        lines, out = random_program(rng, PROG_LEVELS[level][0])
        a = program_ok(lines, out, level)
        if a is not None:
            return lines, out, a


def prog_text(lines, out):
    s = []
    for ln in lines:
        rhs = str(ln["L"]) if ln["kind"] == "lit" else ln["u"] if ln["kind"] == "copy" else \
            f"{ln['u']} {ln['op']} {ln['L']}"
        s.append(f"{ln['v']} = {rhs}")
    return "\n".join(s) + f"\nprint({out})\n# output: "


def story_text(lines, out, obj):
    seen, s = set(), []
    for ln in lines:
        ow = ln["v"] in seen
        seen.add(ln["v"])
        key = ln["kind"] if ln["kind"] != "op" else ln["op"]
        t = STORY_TEMPLATES[key].format(v=PERSON[ln["v"]], u=PERSON.get(ln.get("u"), ""),
                                        L=ln.get("L"), o=obj)
        s.append(("Then " + t) if ow else t)
    return " ".join(s) + f" How many {obj} does {PERSON[out]} have? Answer: "


def arith_text(lines, path):
    expr = None
    for j in path:
        ln = lines[j]
        if ln["kind"] == "lit":
            expr = str(ln["L"])
        elif ln["kind"] == "op":
            expr = f"({expr} {ln['op']} {ln['L']})"
    # drop the outermost brackets
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1]
    return expr + " = "


# ---------------------------------------------------------------- items
def _prog_like(family, level, rng):
    lines, out, a = draw_program(rng, level)
    obj = rng.choice(OBJECTS)
    if family == "PROG":
        text = prog_text(lines, out)
    elif family == "STORY":
        text = story_text(lines, out, obj)
    else:
        text = arith_text(lines, a["path"])
    return {"text": text, "answer": str(a["answer"]), "lines": lines, "out": out,
            "path": a["path"], "distractors": a["distractors"], "stale": a["stale"],
            "values": a["values"], "object": obj if family == "STORY" else None}


def _order(level, rng):
    n = ORDER_LEVELS[level]
    people = rng.sample(NAMES, n)  # tallest first
    pairs = [(people[i], people[i + 1]) for i in range(n - 1)]
    rng.shuffle(pairs)
    q = rng.choice(["shortest", "tallest", "second tallest"])
    ans = {"shortest": people[-1], "tallest": people[0], "second tallest": people[1]}[q]
    text = " ".join(f"{a} is taller than {b}." for a, b in pairs)
    qs = f"Who is the {q}?" if q != "second tallest" else "Who is second tallest?"
    return {"text": f"{text} {qs} Answer:", "answer": " " + ans, "people": people, "question": q}


def _twohop(rng):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "middle"))
    from countries import ROWS  # read-only
    rows = [r for r in ROWS if r[3] in LANGUAGES]
    country, capital, _, lang = rng.choice(rows)
    return {"text": f"The official language of the country whose capital is {capital} is",
            "answer": " " + lang, "country": country, "capital": capital}


def _one(family, level, rng):
    if family in ("PROG", "STORY", "ARITH"):
        return _prog_like(family, level, rng)
    if family == "ORDER":
        return _order(level, rng)
    return _twohop(rng)


def _names_numbers(it, family):
    if family in ("PROG", "STORY", "ARITH"):
        return {it["answer"]}
    if family == "ORDER":
        return set(it["people"])
    return {it["capital"], it["answer"]}


def make(family, level, seed):
    # PROG, STORY and ARITH share the rng key, so the same seed gives twins of one program.
    key = "PROG" if family in ("PROG", "STORY", "ARITH") else family
    rng = random.Random(f"{key}-{level}-{seed}")
    test = _one(family, level, rng)
    shots = []
    while len(shots) < 3:
        ex = _one(family, level, rng)
        if ex["text"] == test["text"] or ex["answer"] == test["answer"]:
            continue
        if family in ("ORDER", "TWOHOP") and _names_numbers(ex, family) & _names_numbers(test, family):
            continue
        if any(ex["text"] == s["text"] for s in shots):
            continue
        shots.append(ex)
    prefix = "".join(f"{s['text']}{s['answer'].strip() if family in ('PROG', 'STORY', 'ARITH') else s['answer']}\n\n"
                     for s in shots)
    return {**test, "family": family, "level": level, "seed": seed,
            "prompt": prefix + test["text"], "shots": [s["text"] + s["answer"] for s in shots]}
