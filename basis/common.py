"""Experiment D shared code (basis/PLAN.md section 1).

Matrices per layer, after folding the RMSNorm gains (repeat/task4b.fold), written with the
residual side first, shape (d, n): q, k, v, gate, up (reading: transposed weight), o, down
(writing: the weight itself).

Own basis: top-k eigenvectors of G = W W^T (left singular vectors of W). Shared basis of a zone:
top-k eigenvectors of S = sum over the zone's layers of G_l. Kept energy of a basis U:
|U U^T W|^2 / |W|^2 = sum_i u_i^T G u_i / tr G. Null: W_l -> Q_l W_l with Q_l a random orthogonal
matrix (seed = layer index), so G_l -> Q_l G_l Q_l^T. Outlier setting: the 16 outlier residual
rows of W set to zero (rows and columns of G zeroed); for the null the rows are zeroed first,
then the layer is rotated.

All linear algebra in float64 on CPU (numpy). Weights are read one layer at a time.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import evaluate as E  # noqa: E402
from repeat import task4b as B4  # noqa: E402

TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
GROUPS = {"attn": ["q", "k", "v", "o"], "mlp": ["gate", "up", "down"]}
SETTINGS = ["all", "no_outliers"]
ZONES = ["ENTRY", "FAMILY", "OUTPUT"]
CACHE = os.path.join(ROOT, "data", "cache", "basis")


def zones(n):
    """ENTRY below 21% of depth, FAMILY 21% to 64%, OUTPUT above (rounded to whole layers)."""
    a, b = round(0.21 * n), round(0.64 * n)
    return {"ENTRY": list(range(0, a)), "FAMILY": list(range(a, b)), "OUTPUT": list(range(b, n))}


def zone_of(l, z):
    return next(k for k, v in z.items() if l in v)


def matrices(W, l):
    """Folded matrices of layer l, residual side first, float64 numpy (d, n)."""
    f = B4.fold(W.raw(l))
    return {"q": f["q"].T, "k": f["k"].T, "v": f["v"].T, "o": f["o"], "gate": f["gate"].T,
            "up": f["up"].T, "down": f["down"]}


def to64(t):
    return t.double().numpy()


def rotation(l, d):
    """Haar-random orthogonal d x d matrix, seed = layer index."""
    rng = np.random.default_rng(l)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))[None, :]


def zero_rows(G, out):
    G = G.copy()
    G[out, :] = 0
    G[:, out] = 0
    return G


def grams(W, l, outliers, d, Q=None):
    """{type: {setting: G}} and, with Q, the rotated grams {type: {setting: Q G Q^T}}."""
    M = matrices(W, l)
    real, rot = {}, {}
    for t in TYPES:
        w = to64(M[t])
        G = w @ w.T
        real[t] = {"all": G, "no_outliers": zero_rows(G, outliers)}
        if Q is not None:
            rot[t] = {s: Q @ real[t][s] @ Q.T for s in SETTINGS}
    del M
    return real, rot


def own_curve(G):
    """Kept energy of the own basis for k = 1..d (cumulative top eigenvalues / trace)."""
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    return np.cumsum(ev) / np.trace(G)


def basis_curve(U, G):
    """Kept energy for k = 1..d of the basis U (columns ordered by importance)."""
    return np.cumsum(np.einsum("ij,ij->j", U, G @ U)) / np.trace(G)


def top_eigvecs(S):
    w, V = np.linalg.eigh(S)
    return V[:, ::-1].copy(), w[::-1].copy()


def outliers_for(model):
    p = {"dev": os.path.join(ROOT, "results", "repeat", "task4b", "dev", "summary.json"),
         "main": os.path.join(ROOT, "results", "repeat", "task4b", "summary.json"),
         "confirm": os.path.join(ROOT, "results", "basis", "task3", "outliers.json")}[model]
    with open(p) as f:
        return np.array(json.load(f)["outliers"])


def weights(model):
    return B4.Weights(E.MODELS[model])


def shared_bases(W, layers_by_zone, outliers, rotate=True, log=print):
    """Pass 1: own-energy curves per layer, and the eigenvectors of each zone's summed grams
    (real and rotated null), per type and setting. One zone's sums are held at a time."""
    d = W.D
    own, U, Un = {}, {}, {}
    for z, layers in layers_by_zone.items():
        S = {t: {s: np.zeros((d, d)) for s in SETTINGS} for t in TYPES}
        Sn = {t: {s: np.zeros((d, d)) for s in SETTINGS} for t in TYPES} if rotate else None
        for l in layers:
            Q = rotation(l, d) if rotate else None
            real, rot = grams(W, l, outliers, d, Q)
            for t in TYPES:
                for s in SETTINGS:
                    S[t][s] += real[t][s]
                    if rotate:
                        Sn[t][s] += rot[t][s]
                    own[(l, t, s)] = own_curve(real[t][s])
            del real, rot
            log(f"pass 1 layer {l} ({z})")
        for t in TYPES:
            for s in SETTINGS:
                U[(z, t, s)] = top_eigvecs(S[t][s])[0]
                if rotate:
                    Un[(z, t, s)] = top_eigvecs(Sn[t][s])[0]
        del S, Sn
        log(f"eigenvectors {z}")
    return own, U, Un


def energies(W, layers_by_zone, outliers, own, U, Un, cross_zone="FAMILY", log=print):
    """Pass 2: kept-energy curves (k = 1..d) per layer, type and setting:
    own, shared (own zone), null (rotated layer, rotated-null basis of its zone), null_own
    (own basis of the rotated layer, for the invariance check), and for the cross zone the other
    zones' shared bases."""
    d = W.D
    out = {}
    for z, layers in layers_by_zone.items():
        for l in layers:
            Q = rotation(l, d) if Un else None
            real, rot = grams(W, l, outliers, d, Q)
            for t in TYPES:
                for s in SETTINGS:
                    G = real[t][s]
                    r = {"own": own[(l, t, s)], "shared": basis_curve(U[(z, t, s)], G)}
                    if Un:
                        Gr = rot[t][s]
                        r["null"] = basis_curve(Un[(z, t, s)], Gr)
                        r["null_own"] = own_curve(Gr)
                    if z == cross_zone:
                        for z2 in layers_by_zone:
                            if z2 != z and (z2, t, s) in U:
                                r[f"basis_{z2}"] = basis_curve(U[(z2, t, s)], G)
                    out[(z, l, t, s)] = r
            del real, rot
            log(f"pass 2 layer {l} ({z})")
    return out
