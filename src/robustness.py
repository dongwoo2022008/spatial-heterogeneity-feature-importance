"""Section 4.5. Heavy; skipped by --core."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from .config import FEATURES, TARGET, CATBOOST, STAGES, SEED, N_BOOT_GLOBAL, OUT_T
from . import model as M
from .checks import check

def run_all(S, m, A, cc):
    ok = []
    train, hold = S["train"], S["holdout"]
    # (2) global rank bootstrap with refitting
    rng = np.random.default_rng(SEED)
    B = np.zeros((N_BOOT_GLOBAL, len(FEATURES)), int)
    for b in range(N_BOOT_GLOBAL):
        s = train.sample(len(train), replace=True, random_state=b)
        mb = M.fit(s)
        B[b] = (-np.abs(M.shap_matrix(mb, cc)).mean(0)).argsort().argsort() + 1
    med = np.median(B, 0)
    pd.DataFrame({"predictor": FEATURES, "median_rank": med.astype(int),
                  "lo": np.percentile(B, 2.5, 0).astype(int),
                  "hi": np.percentile(B, 97.5, 0).astype(int)}).to_csv(
                  OUT_T / "tableA6_global_rank_bootstrap.csv", index=False)
    ok.append(check("PageRank median global rank", int(med[FEATURES.index("pagerank_lag1")]), 5))
    # (3) leave-one-province-out
    imp = S["imputed"]
    rows = []
    for s in sorted(imp.sido.unique()):
        te = imp[imp.sido == s]; tr = imp[imp.sido != s]
        if len(te) < 10: continue
        mb = M.fit(tr)
        rows.append((int(s), len(te), r2_score(te[TARGET], mb.predict(te[FEATURES]))))
    cv = pd.DataFrame(rows, columns=["sido", "n", "r2"])
    cv.to_csv(OUT_T / "tableA7_spatial_cv.csv", index=False)
    ok.append(check("spatial CV mean R2 (rounded)", round(float(cv.r2.mean()), 2), 0.05, tol=0.02))
    return ok
