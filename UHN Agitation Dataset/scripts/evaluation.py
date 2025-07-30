
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_and_print(true, preds, model_name="Model"):
    precision = precision_score(true, preds, zero_division=0)
    recall = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)
    auc = roc_auc_score(true, preds)
    print(f"{model_name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
