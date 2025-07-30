
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import hdbscan

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def run_hdbscan(X, min_cluster_size=10):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(X)

def run_one_class_svm(X, nu=0.05, kernel='rbf', gamma='scale'):
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    return model.fit_predict(X)
