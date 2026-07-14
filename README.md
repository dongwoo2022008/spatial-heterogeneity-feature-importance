# Interpreting feature importance under spatial heterogeneity

Replication package for the paper on urban development stages and the interpretation of
SHAP feature importance in regional migration analysis (South Korea, 229 municipalities,
2017–2024).

## Reproduce

```bash
pip install -r requirements.txt
python reproduce.py --core     # canonical model + inference  (~3 min)
python reproduce.py            # + robustness analyses        (~15 min)
```

`reproduce.py` regenerates every table, figure and in-text number, then **asserts them against
the values printed in the manuscript** (`src/checks.py`). If a line reads `FAIL`, code and paper
have diverged. There is no other source of truth.

## The two-sample design (read this before changing anything)

| purpose | sample | n |
|---|---|---|
| model selection, estimation, hold-out evaluation | full panel, year-median imputation | train 1,374 / hold-out 458 |
| **SHAP interpretation** | **complete cases only** | **1,480** |

Fitting and interpretation deliberately use different samples. All municipalities inform the
model, but **no imputed value is ever assigned a feature attribution**, and the global and the
stage-stratified attributions rest on one basis, so their ranks are directly comparable.

This is not cosmetic:

* changing the **training** sample to complete cases only moves the global ranking (ρ = 0.82;
  sewerage supply 4 → 8) and flips model selection to XGBoost;
* changing the **evaluation** sample leaves it essentially unchanged (ρ = 0.99).

Both sensitivities are reported in the paper (Section 4.5).

## Inference

Stage dependence of the importance ranking is tested, not asserted.

* **Cluster bootstrap** (2,000 replications) resamples *municipalities*, not municipality-years,
  so the panel structure is preserved → 95% CI for within-stage ranks.
* **Permutation test** (2,000 permutations) reassigns development-stage labels across
  municipalities → predictor-specific null distribution of the rank range.
  H0: a predictor's importance rank is invariant across spatial regimes.
* **Benjamini–Hochberg FDR** across the twenty simultaneous tests; significance reported at q < 0.05.

Both procedures hold the fitted model fixed. Estimation variability is handled separately by a
bootstrap **with model refitting** (50 replications, Section 4.5).

## Headline values asserted by `reproduce.py`

| | |
|---|---|
| global top-5 (mean \|SHAP\|, ‰) | childcare 2.59 · housing age 1.99 · closeness 1.87 · sewerage 1.85 · PageRank 1.59 |
| PageRank stage ranks (low / int / high) | **1 / 6 / 9**  (q = 0.002) |
| Childcare stage ranks | 2 / 1 / 1  (rank range 1) |
| stage-dependent predictors | **12 of 20** at q < 0.05 |
| signed mean SHAP, PageRank | **−1.70 / +0.72 / +1.18** ‰ |
| signed mean SHAP, childcare | **+0.33 / −0.91 / −1.26** ‰ |
| CatBoost hold-out R² | 0.324 |

Two facts the ranking alone conceals, and which the paper is about: PageRank is the *most*
informative predictor where municipalities are sparsest, yet its contribution there is
*negative*; childcare is the most important predictor everywhere, yet the *sign* of its
contribution reverses across the settlement hierarchy.

## Layout

```
data/panel_2017_2024.csv     229 municipalities × 8 years (1,832 rows)
src/config.py                seeds, hyperparameters, the two samples
src/data.py                  loading, imputation, stage assignment
src/model.py                 the single fitted model and its SHAP matrix
src/inference.py             cluster bootstrap · permutation test · BH-FDR
src/robustness.py            Section 4.5
src/checks.py                the manuscript's numbers
reproduce.py                 entry point
outputs/                     generated (not committed)
```

## Data

Municipality-level panel compiled from Statistics Korea (KOSIS) resident-registration and
internal-migration statistics and related official series. PageRank and closeness centrality are
computed on annual directed, weighted municipality-to-municipality migration networks
(NetworkX); PageRank enters lagged one year to avoid target leakage.
