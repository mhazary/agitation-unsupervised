# UHN Agitation Dataset: Unsupervised Detection of Agitation

This repository contains the folder structure and framework for detecting agitation in dementia patients using unsupervised learning techniques. The dataset includes 20 participant folders, each representing sensor data collected during clinical observations.

## Project Overview

We apply unsupervised models to identify agitation episodes based on physiological and movement sensor data. This approach supports scalable, non-invasive monitoring in healthcare settings.

## Modules

- `preprocessing.py`: Load and normalize data  
- `dimensionality_reduction.py`: PCA logic  
- `clustering_models.py`: DBSCAN, HDBSCAN, One-Class SVM  
- `anomaly_models.py`: Isolation Forest, Autoencoder (add later)  
- `hybrid_models.py`: Ensemble of models  
- `evaluation.py`: Metrics evaluation  
- `oversampling.py`: SMOTE oversampling  

## Installation

```bash
pip install -r requirements.txt