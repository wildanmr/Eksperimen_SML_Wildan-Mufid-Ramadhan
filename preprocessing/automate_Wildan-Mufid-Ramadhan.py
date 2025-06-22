import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Menangani Missing Values (jika ada)
    df.dropna(inplace=True)
    # Menangani Duplikasi Data
    df.drop_duplicates(inplace=True)
    # Normalisasi/Standarisasi Fitur
    scaler = StandardScaler()
    numerical_cols = [col for col in df.columns if df[col].nunique() > 2 and col != 'Diabetes_binary']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

if __name__ == "__main__":
    df_raw = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df_preprocessed = preprocess_data(df_raw)
    df_preprocessed.to_csv("preprocessing/diabetes_preprocessed.csv", index=False)
    print("Data preprocessing complete and saved to diabetes_preprocessed.csv")

