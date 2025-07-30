
import numpy as np

def hybrid_and_condition(dbscan_labels, iforest_scores, threshold=-0.1):
    iforest_anomaly = (iforest_scores < threshold).astype(int)
    dbscan_anomaly = (dbscan_labels == -1).astype(int)
    hybrid = (iforest_anomaly & dbscan_anomaly)
    return hybrid

def apply_threshold(scores, quantile=0.05):
    threshold = np.quantile(scores, quantile)
    return (scores < threshold).astype(int), threshold
