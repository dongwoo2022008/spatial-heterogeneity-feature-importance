"""Inference on stage-dependence of the importance ranking.

Two procedures, both holding the fitted model FIXED:
  1. cluster bootstrap  -> 95% CI for within-stage ranks (sampling variability)
  2. label permutation  -> null distribution of the rank range (H0: rank invariant to stage)
Benjamini-Hochberg FDR across the 20 simultaneous tests.
"""
import numpy as np, pandas as pd
from .config import FEATURES, LABELS, STAGES, N_BOOT_STAGE, N_PERM, SEED

def stage_ranks(abs_shap, stage, idx=None):
    out = {}
    for s in STAGES:
        m = (stage == s) if idx is None else (stage[idx] == s)
        sub = abs_shap[idx][m] if idx is not None else abs_shap[m]
        out[s] = (-sub.mean(0)).argsort().argsort() + 1
    return out

def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); ps = p[o]
    q = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out

def run(abs_shap, cc):
    stage = cc["stage"].values
    reg = cc["region_code"].values
    obs = stage_ranks(abs_shap, stage)
    rng = np.random.default_rng(SEED)

    regs = np.unique(reg); rid = {r: np.where(reg == r)[0] for r in regs}
    boot = np.zeros((N_BOOT_STAGE, 3, len(FEATURES)), int)
    for b in range(N_BOOT_STAGE):
        idx = np.concatenate([rid[r] for r in rng.choice(regs, len(regs), replace=True)])
        r = stage_ranks(abs_shap, stage, idx)
        for j, s in enumerate(STAGES):
            boot[b, j] = r[s]

    reg_stage = cc.groupby("region_code")["stage"].first()
    null = np.zeros((N_PERM, len(FEATURES)))
    for p in range(N_PERM):
        pm = dict(zip(reg_stage.index, rng.permutation(reg_stage.values)))
        ps = np.array([pm[r] for r in reg])
        rr = np.vstack([stage_ranks(abs_shap, ps)[s] for s in STAGES])
        null[p] = rr.max(0) - rr.min(0)

    g = abs_shap.mean(0); grank = (-g).argsort().argsort() + 1
    rows = []
    for i, f in enumerate(FEATURES):
        o = [int(obs[s][i]) for s in STAGES]
        rr = max(o) - min(o)
        pv = float((null[:, i] >= rr).mean())
        pv = max(pv, 1 / N_PERM)
        rows.append(dict(
            predictor=LABELS[f], global_rank=int(grank[i]), mean_abs_shap=round(float(g[i]), 3),
            low=o[0], low_ci=f"{int(np.percentile(boot[:,0,i],2.5))}-{int(np.percentile(boot[:,0,i],97.5))}",
            intermediate=o[1], int_ci=f"{int(np.percentile(boot[:,1,i],2.5))}-{int(np.percentile(boot[:,1,i],97.5))}",
            high=o[2], high_ci=f"{int(np.percentile(boot[:,2,i],2.5))}-{int(np.percentile(boot[:,2,i],97.5))}",
            rank_range=rr, perm_p=round(pv, 4)))
    out = pd.DataFrame(rows)
    out["fdr_q"] = np.round(bh(out.perm_p.values), 4)
    return out.sort_values("global_rank").reset_index(drop=True)
