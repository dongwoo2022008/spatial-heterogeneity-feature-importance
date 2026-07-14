"""Loading and the two canonical samples."""
import numpy as np, pandas as pd
from .config import DATA, FEATURES, TARGET, YEAR_TRAIN_END, STAGES

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA).dropna(subset=[TARGET]).copy()
    df["sido"] = (df["region_code"] // 1000).astype(int)
    return df

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Year-specific median imputation. Used for MODEL FITTING ONLY."""
    out = df.copy()
    for f in FEATURES:
        out[f] = out.groupby("year")[f].transform(lambda s: s.fillna(s.median()))
    return out

def assign_stage(df: pd.DataFrame) -> pd.Series:
    """Density tertiles computed on municipality means over the FULL panel (229 units)."""
    tert = pd.qcut(df.groupby("region_code")["pop_density"].mean(), 3, labels=STAGES)
    return df["region_code"].map(tert.to_dict())

def samples():
    raw = load_raw()
    imp = impute(raw)
    stage = assign_stage(raw)
    imp["stage"] = stage
    train = imp[imp.year <= YEAR_TRAIN_END]           # 1,374 rows
    holdout = imp[imp.year > YEAR_TRAIN_END]          # 458 rows
    cc = raw.dropna(subset=FEATURES).copy()           # complete cases
    cc["stage"] = assign_stage(raw).loc[cc.index]
    cc = cc.dropna(subset=["stage"]).reset_index(drop=True)   # 1,480 rows
    return dict(raw=raw, imputed=imp, train=train, holdout=holdout, interpret=cc)
