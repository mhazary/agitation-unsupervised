
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_participant_csv(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

def normalize_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df
