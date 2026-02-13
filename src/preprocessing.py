import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_sample_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Time"])

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0].sample(n=20000, random_state=42)

    df_sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

    X = df_sample.drop("Class", axis=1)
    y = df_sample["Class"]

    return X, y, scaler