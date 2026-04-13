"""
Script to take loaded data and preprocess for modeling.
Usage in other scripts / notebooks:
    from preprocessor import preprocess_data
    X_train, X_test, y_train, y_test = preprocess(df)

- Scales Amount and Time features
- Stratified split: 70/15/15
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    #Scaling Amount and Time features
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    #Separating features and target variable
    features = [col for col in df.columns if col != "Class"]
    X = df[features].values
    y = df["Class"].values

    #Stratified split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Split sizes: Train={len(X_train):,}, Validation={len(X_val):,}, Test={len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test