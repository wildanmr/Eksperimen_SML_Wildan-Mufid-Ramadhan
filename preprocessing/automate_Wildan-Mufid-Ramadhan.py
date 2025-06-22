import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Menangani Missing Values (jika ada)
    df.dropna(inplace=True)
    # Menangani Duplikasi Data
    df.drop_duplicates(inplace=True)
    # Normalisasi/Standarisasi Fitur
    target_column = 'Diabetes_binary'
    if target_column in df.columns:
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Combine scaled features with target
        df = pd.concat([X_scaled, y], axis=1)
        
        print("Features have been standardized.")
    else:
        raise(f"Target column '{target_column}' not found. Please adjust the target_column variable.\nAvailable columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df_raw = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df_preprocessed = preprocess_data(df_raw)
    df_preprocessed.to_csv("preprocessing/diabetes_preprocessed.csv", index=False)
    print("Data preprocessing complete and saved to diabetes_preprocessed.csv")

