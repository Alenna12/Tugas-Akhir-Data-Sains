import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "Dengue_Climate_Bangladesh - DengueAndClimateBangladesh.csv"
FEATURES = ["YEAR", "MONTH", "MIN", "MAX", "HUMIDITY", "RAINFALL"]
TARGET = "DENGUE"

TEST_SIZE = 0.2
RANDOM_SEED = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df.dropna(subset=required).copy()
    return df


def train_regression(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2


def plot_rainfall_vs_dengue(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    plt.scatter(df["RAINFALL"], df[TARGET])
    plt.xlabel("Curah Hujan (RAINFALL)")
    plt.ylabel("Kasus DBD (DENGUE)")
    plt.title("Hubungan RAINFALL vs DENGUE")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


def plot_actual_vs_pred(y_test, y_pred):
    plt.figure(figsize=(9, 5))
    plt.scatter(y_test, y_pred)

    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Nilai Aktual (DENGUE)")
    plt.ylabel("Nilai Prediksi (DENGUE)")
    plt.title("Perbandingan Aktual vs Prediksi (Regresi Linear)")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


def predict_new(model: LinearRegression) -> float:
    data_baru = pd.DataFrame([{
        "YEAR": 2020,
        "MONTH": 6,
        "MIN": 25.0,
        "MAX": 33.0,
        "HUMIDITY": 80.0,
        "RAINFALL": 200.0
    }])

    return float(model.predict(data_baru)[0])


def main():
    df = load_data(DATA_PATH)

    print("Jumlah data (setelah dropna):", len(df))
    print("Kolom:", list(df.columns))

    model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2 = train_regression(df)

    print("\nPembagian data:")
    print("Data latih:", X_train.shape[0])
    print("Data uji  :", X_test.shape[0])

    print("\nEvaluasi model:")
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

    print("\nKoefisien:")
    for f, c in zip(FEATURES, model.coef_):
        print(f"{f}: {c:.6f}")
    print("Intercept:", f"{model.intercept_:.6f}")

    pred_baru = predict_new(model)
    print("\nContoh prediksi data baru:")
    print("Prediksi jumlah kasus DENGUE:", pred_baru)

    plot_rainfall_vs_dengue(df)
    plot_actual_vs_pred(y_test, y_pred)


if __name__ == "__main__":
    main()