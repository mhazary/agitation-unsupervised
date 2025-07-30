
from sklearn.ensemble import IsolationForest
import numpy as np

def run_isolation_forest(X, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)
    return preds, scores
