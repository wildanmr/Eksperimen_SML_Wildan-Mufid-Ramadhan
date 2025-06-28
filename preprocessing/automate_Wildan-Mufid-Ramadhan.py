import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

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
        
        # Save the scaler model
        os.makedirs("preprocessing", exist_ok=True)
        joblib.dump(scaler, "preprocessing/scaler.pkl")
        print("Scaler model saved to preprocessing/scaler.pkl")
        
        # Combine scaled features with target
        df = pd.concat([X_scaled, y], axis=1)
        
        print("Features have been standardized.")
    else:
        raise ValueError(f"Target column '{target_column}' not found. Please adjust the target_column variable.\nAvailable columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df_raw = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df_preprocessed = preprocess_data(df_raw)
    
    os.makedirs("preprocessing", exist_ok=True)
    df_preprocessed.to_csv("preprocessing/diabetes_preprocessed.csv", index=False)
    print("Data preprocessing complete and saved to diabetes_preprocessed.csv")
    print("Scaler model can be loaded later using: scaler = joblib.load('preprocessing/scaler.pkl')")