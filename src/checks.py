"""Values printed in the manuscript. reproduce.py asserts against these."""
EXPECTED = {
    "n_train": 1374, "n_holdout": 458, "n_interpret": 1480,
    "stage_n": {"low": 588, "intermediate": 528, "high": 364},
    "global_top5": ["Childcare facilities", "Housing age", "Closeness centrality",
                    "Sewerage supply", "PageRank (t-1)"],
    "global_shap_top5": [2.59, 1.99, 1.87, 1.85, 1.59],
    "pagerank_stage_ranks": [1, 6, 9],
    "childcare_stage_ranks": [2, 1, 1],
    "n_significant_fdr": 12,
    "mean_signed_shap": {          # Section 4.3.4
        "PageRank (t-1)": [-1.70, 0.72, 1.18],
        "Childcare facilities": [0.33, -0.91, -1.26],
    },
    "catboost_holdout_r2": 0.324,
}
def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and tol else (got == want)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {got}" + ("" if ok else f"  (expected {want})"))
    return ok
