"""Deliberately leaky pipeline used to verify tools/audit.py actually detects leakage.

Every construct here is a real mistake that produces an inflated score with no error and
no metric anomaly. This file is never executed; it exists so the audit tool can be tested
against code that is known to be wrong.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 0


def build():
    # LEAK 1 — synthetic data presented as if it were a real source.
    X, y = make_classification(n_samples=1000, n_features=20, random_state=SEED)

    # LEAK 2 — scaler fitted on the full dataset before splitting, so test-set statistics
    # enter the training transform.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame(X)
    df["target"] = y

    # LEAK 3 — rolling window with no preceding shift: the target enters its own feature.
    df["rolling_target"] = df["target"].rolling(7).mean()

    # LEAK 4 — centered window spans future observations outright.
    df["centered"] = df["target"].rolling(7, center=True).mean()

    # LEAK 5 — hard-coded URL bypassing the declared registry, so nothing is hash-pinned.
    extra = pd.read_csv("https://raw.githubusercontent.com/example/repo/main/x.csv")

    return train_test_split(df, test_size=0.2, shuffle=True, random_state=SEED), extra
