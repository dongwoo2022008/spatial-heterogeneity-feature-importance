#!/usr/bin/env python
"""Single entry point. Regenerates every number, table and figure in the paper.

    python reproduce.py            # full pipeline (~15 min)
    python reproduce.py --core     # canonical model + inference only (~3 min)

Every value printed under CHECKS corresponds to a number in the manuscript.
If any line reads FAIL, the manuscript and the code have diverged.
"""
import argparse, sys, numpy as np, pandas as pd
from src.config import FEATURES, LABELS, STAGES, OUT_T, OUT_F, TARGET
from src import data as D, model as M, inference as I
from src.checks import EXPECTED as E, check

def main(core=False):
    OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
    S = D.samples()
    train, cc = S["train"], S["interpret"]
    print(f"samples: train={len(train)}  hold-out={len(S['holdout'])}  interpret={len(cc)}")

    m = M.fit(train)
    shap = M.shap_matrix(m, cc)          # signed
    A = np.abs(shap)
    rank, imp = M.ranks(A)

    # ---- Table 5 : global importance --------------------------------------
    from scipy.stats import spearmanr
    direction = ["+" if spearmanr(cc[f], shap[:, i]).correlation > .1 else
                 ("-" if spearmanr(cc[f], shap[:, i]).correlation < -.1 else "mixed")
                 for i, f in enumerate(FEATURES)]
    t5 = pd.DataFrame({"rank": rank, "predictor": [LABELS[f] for f in FEATURES],
                       "mean_abs_shap": imp.round(2), "direction": direction}).sort_values("rank")
    t5.to_csv(OUT_T / "table5_global_importance.csv", index=False)

    # ---- Tables 6 & A10 : stage inference ---------------------------------
    inf = I.run(A, cc)
    inf.to_csv(OUT_T / "tableA10_stage_rank_inference.csv", index=False)
    focal = ["Childcare facilities", "Housing age", "Closeness centrality",
             "Sewerage supply", "PageRank (t-1)"]
    inf[inf.predictor.isin(focal)].to_csv(OUT_T / "table6_focal_stages.csv", index=False)

    # ---- Section 4.3.4 : signed mean contribution by stage ------------------
    sg = cc["stage"].values
    signed = pd.DataFrame({LABELS[f]: [round(float(shap[sg == s, i].mean()), 2) for s in STAGES]
                           for i, f in enumerate(FEATURES)}, index=STAGES).T
    signed.to_csv(OUT_T / "signed_mean_shap_by_stage.csv")

    # ---- CHECKS ------------------------------------------------------------
    print("\nCHECKS")
    ok = []
    ok.append(check("n_train", len(train), E["n_train"]))
    ok.append(check("n_holdout", len(S["holdout"]), E["n_holdout"]))
    ok.append(check("n_interpret", len(cc), E["n_interpret"]))
    ok.append(check("stage n", dict(cc.stage.value_counts()[STAGES]), E["stage_n"]))
    ok.append(check("global top-5", t5.predictor.head(5).tolist(), E["global_top5"]))
    ok.append(check("global |SHAP| top-5", t5.mean_abs_shap.head(5).round(2).tolist(),
                    E["global_shap_top5"]))
    pr = inf[inf.predictor == "PageRank (t-1)"].iloc[0]
    ok.append(check("PageRank stage ranks", [int(pr.low), int(pr.intermediate), int(pr.high)],
                    E["pagerank_stage_ranks"]))
    ch = inf[inf.predictor == "Childcare facilities"].iloc[0]
    ok.append(check("Childcare stage ranks", [int(ch.low), int(ch.intermediate), int(ch.high)],
                    E["childcare_stage_ranks"]))
    ok.append(check("significant at FDR 5%", int((inf.fdr_q < .05).sum()), E["n_significant_fdr"]))
    for k, v in E["mean_signed_shap"].items():
        ok.append(check(f"signed SHAP {k}", signed.loc[k].tolist(), v))

    if not core:
        from src.robustness import run_all
        ok += run_all(S, m, A, cc)

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true")
    sys.exit(main(**vars(ap.parse_args())))
