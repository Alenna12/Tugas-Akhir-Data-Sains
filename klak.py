import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = "dataset.csv"
TARGET = "Outcome"
K = 5
TEST_SIZE = 0.2
RANDOM_SEED = 42

def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")

    print("Shape:", df.shape)
    print(df.head(5).to_string(index=False))

    X_raw = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_train = pd.get_dummies(X_train_raw, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    print("\nTrain:", X_train.shape, "Test:", X_test.shape)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    one_raw = X_test_raw.iloc[[0]].copy()
    one_enc = pd.get_dummies(one_raw, drop_first=True)
    one_enc = one_enc.reindex(columns=X_train.columns, fill_value=0)
    one_sc = scaler.transform(one_enc)

    pred_one = int(model.predict(one_sc)[0])
    proba_one = float(model.predict_proba(one_sc)[0, 1])

    print("\nContoh 1 data uji (raw):")
    print(one_raw.to_string(index=False))
    print("Prediksi Outcome:", pred_one)
    print("Probabilitas Outcome=1:", round(proba_one, 6))

if __name__ == "__main__":
    main()