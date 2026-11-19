"""Diagnostic for the STOP after T3 (not a gate, changes nothing): why does the evaluation find
~60% dead units when training counts ~2-10% dead in the last 10M positions?
Counts units never active on (a) the gate's held-out data (1,000,192 STORY positions + FACT prompts
of people 0..999), and (b) the same plus held-out biographies of all the run's trained people
(1,954 packed rows, seed 12345, as task 4 uses). Also: (c) units active only on biographies of
people >= 1000, whom the FACT prompts do not cover.

  python binding/diag_dead.py --run P2-main-2k-s0 --seed 0
"""
import argparse, json, os, shlex, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binding import common as B  # noqa: E402
from binding.task2 import Capture, fact_rows  # noqa: E402


@torch.no_grad()
def counts(model, tcs, n_units, rows, mask=None, bs=64):
    c = torch.zeros(B.N_LAYER, n_units, device=B.DEVICE)
    for i in range(0, len(rows), bs):
        with Capture(model) as cap:
            model(torch.from_numpy(rows[i:i + bs]).to(B.DEVICE))
        for l, tc in enumerate(tcs):
            h = cap.hs[l] if mask is None else cap.hs[l][torch.from_numpy(mask[i:i + bs]).to(B.DEVICE)]
            v, u = tc.encode(h.reshape(-1, B.D_MODEL))
            c[l].index_add_(0, u[v > 0], torch.ones_like(v[v > 0]))
    return c.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--tc", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = B.out_dir("diag_dead", a.run)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    model = B.load(a.run)
    tcs, rec = B.load_transcoders(a.run, a.tc)
    n = rec["n_units"]
    story = counts(model, tcs, n, B.heldout_story_rows())
    fx, fm = fact_rows(B.meta(a.run))
    fact = counts(model, tcs, n, fx, fm, bs=512)
    tp = B.trained_people(a.run)
    rng = np.random.default_rng(12345)
    t5 = B.meta(a.run).get("template5_even")
    lo = B.pack_bios(tp[:1000], 977, rng, t5, list(range(1000)))
    hi = B.pack_bios(tp[1000:], 977, rng, t5, list(range(1000, len(tp))))
    bio_lo = counts(model, tcs, n, lo)
    bio_hi = counts(model, tcs, n, hi)
    gate_dead = (story == 0) & (fact == 0)
    tot = story + fact
    res = {"run": a.run, "tc": a.tc, "recipe": rec["name"], "n_units": n,
           "dead_gate_data": gate_dead.mean(1).tolist(),
           "dead_gate_data_plus_bios_all_people": (gate_dead & (bio_lo == 0) & (bio_hi == 0)).mean(1).tolist(),
           "gate_dead_but_active_on_bios_people_0_999": (gate_dead & (bio_lo > 0)).mean(1).tolist(),
           "gate_dead_but_active_only_on_bios_people_ge_1000": (gate_dead & (bio_lo == 0) & (bio_hi > 0)).mean(1).tolist(),
           "units_by_gate_data_count": {
               k: [float(((tot[l] >= lo_) & (tot[l] <= hi_)).mean()) for l in range(B.N_LAYER)]
               for k, (lo_, hi_) in {"0": (0, 0), "1": (1, 1), "2-3": (2, 3), "4-10": (4, 10),
                                     "11-100": (11, 100), "101-1000": (101, 1000),
                                     ">1000": (1001, np.inf)}.items()},
           "positions": {"story": int(B.heldout_story_rows().size), "fact": int(fm.sum()),
                         "bios_people_0_999": int(lo.size), "bios_people_ge_1000": int(hi.size)}}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
