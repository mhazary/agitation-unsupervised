
from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
