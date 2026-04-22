import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)

    # Drop label for training (unsupervised)
    X = df.drop("Class", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df["Class"]