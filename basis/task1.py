"""Basis task 1: how much is shared (basis/PLAN.md). Weights only.

For each zone, matrix type, outlier setting and k = d/16, d/8, d/4, d/2: mean kept energy with
the layers' own bases, with the zone's shared basis and (rotated layers) with the rotated-null
shared basis; sharing efficiency (mean over the zone's layers of shared / own) for real and null.
For the FAMILY also the other zones' shared bases applied to family layers.
Saves the FAMILY shared and null bases (all coordinates) to data/cache/basis/<model>/ for task 2.

  python basis/task1.py --model main --seed 0          (task 3 uses --model confirm)
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basis import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def out_dir(model, task):
    d = os.path.join(C.ROOT, "results", "basis", task)
    return d if model in ("main", "confirm") else os.path.join(d, model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dev", "main", "confirm"], default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    t0 = time.time()
    task = "task3" if args.model == "confirm" else "task1"
    out = out_dir(args.model, task)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    W = C.weights(args.model)
    d, z = W.D, C.zones(W.n_layers)
    ks = [d // 16, d // 8, d // 4, d // 2]
    outl = C.outliers_for(args.model)
    log = lambda m: print(f"{time.time() - t0:.0f}s {m}", flush=True)  # noqa: E731
    own, U, Un = C.shared_bases(W, z, outl, log=log)
    cache = os.path.join(C.CACHE, args.model)
    os.makedirs(cache, exist_ok=True)
    for t in C.TYPES:
        np.save(os.path.join(cache, f"FAMILY_{t}_shared.npy"), U[("FAMILY", t, "all")][:, :d // 2])
        np.save(os.path.join(cache, f"FAMILY_{t}_null.npy"), Un[("FAMILY", t, "all")][:, :d // 2])
    curves = C.energies(W, z, outl, own, U, Un, log=log, null_own=False)
    del U, Un
    rows, per_layer = [], {}
    for zone in C.ZONES:
        for t in C.TYPES:
            for s in C.SETTINGS:
                rs = [curves[(zone, l, t, s)] for l in z[zone]]
                for k in ks:
                    row = {"zone": zone, "type": t, "setting": s, "k": k, "k_frac": k / d,
                           "basis": "own zone",
                           "own": float(np.mean([r["own"][k - 1] for r in rs])),
                           "shared": float(np.mean([r["shared"][k - 1] for r in rs])),
                           "null_shared": float(np.mean([r["null"][k - 1] for r in rs])),
                           "eff_real": float(np.mean([r["shared"][k - 1] / r["own"][k - 1]
                                                      for r in rs])),
                           "eff_null": float(np.mean([r["null"][k - 1] / r["own"][k - 1]
                                                      for r in rs]))}
                    rows.append(row)
                    if zone == "FAMILY":
                        for z2 in C.ZONES:
                            if z2 != zone:
                                rows.append({**row, "basis": z2,
                                             "shared": float(np.mean([r[f"basis_{z2}"][k - 1]
                                                                      for r in rs])),
                                             "eff_real": float(np.mean(
                                                 [r[f"basis_{z2}"][k - 1] / r["own"][k - 1]
                                                  for r in rs])),
                                             "null_shared": "", "eff_null": ""})
                for l in z[zone]:
                    r = curves[(zone, l, t, s)]
                    per_layer[f"{zone}/{l}/{t}/{s}"] = {
                        name: [float(r[name][k - 1]) for k in ks] for name in r}
    with open(os.path.join(out, "energy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": C.E.MODELS[args.model], "seed": args.seed, "d": d, "ks": ks, "zones": z,
           "outliers": outl.tolist(), "per_layer_at_ks": per_layer,
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f)
    plots(rows, out, d)
    log("done")


def plots(rows, out, d):
    for t in C.TYPES:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for ax, s in zip(axs, C.SETTINGS):
            for i, zone in enumerate(C.ZONES):
                r = [x for x in rows if x["zone"] == zone and x["type"] == t and x["setting"] == s
                     and x["basis"] == "own zone"]
                kf = [x["k_frac"] for x in r]
                ax.plot(kf, [x["eff_real"] for x in r], "-o", color=f"C{i}", label=f"{zone} real")
                ax.plot(kf, [x["eff_null"] for x in r], "--o", color=f"C{i}", alpha=0.6,
                        label=f"{zone} null")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k / d")
            ax.set_ylabel("sharing efficiency (shared / own kept energy)")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{t} [{s}]")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"efficiency_{t}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
