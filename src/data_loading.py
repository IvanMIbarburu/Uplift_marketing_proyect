import os
import pandas as pd
import joblib
import numpy as np

# --- Configuración de rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "criteo-uplift-v2.1.csv")
MODEL_TREATMENT_PATH = os.path.join(BASE_DIR, "..", "notebooks", "model_treatment.joblib")
MODEL_CONTROL_PATH   = os.path.join(BASE_DIR, "..", "notebooks", "model_control.joblib")
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "processed",
    "criteo-uplift-processed_sample20_sorted.parquet"
)

FEATURE_COLS = [f"f{i}" for i in range(12)]


def prepare_processed_data(sample_frac=0.2, random_state=42):

    print("Cargando datos crudos...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Cargando modelos...")
    model_treat = joblib.load(MODEL_TREATMENT_PATH)
    model_ctrl  = joblib.load(MODEL_CONTROL_PATH)

    print("Calculando probabilidades...")
    df["p_treatment"] = model_treat.predict_proba(df[FEATURE_COLS])[:, 1]
    df["p_control"]   = model_ctrl.predict_proba(df[FEATURE_COLS])[:, 1]
    df["uplift_score"] = df["p_treatment"] - df["p_control"]

    print("Muestreo estratificado 20%...")
    df_sample = (
        df.groupby("treatment", group_keys=False)
          .apply(lambda x: x.sample(frac=sample_frac, random_state=random_state))
          .reset_index(drop=True)
    )

    print("Ordenando por uplift_score...")
    df_sample = df_sample.sort_values("uplift_score", ascending=False).reset_index(drop=True)

    print("Calculando métricas acumuladas...")
    df_sample["cum_treated"] = (df_sample["treatment"] == 1).cumsum()
    df_sample["cum_control"] = (df_sample["treatment"] == 0).cumsum()

    df_sample["cum_conv_treated"] = (
        df_sample["conversion"] * (df_sample["treatment"] == 1)
    ).cumsum()

    df_sample["cum_conv_control"] = (
        df_sample["conversion"] * (df_sample["treatment"] == 0)
    ).cumsum()

    df_sample["treat_rate"] = df_sample["cum_conv_treated"] / df_sample["cum_treated"].replace(0, np.nan)
    df_sample["control_rate"] = df_sample["cum_conv_control"] / df_sample["cum_control"].replace(0, np.nan)

    df_sample["observed_uplift"] = df_sample["treat_rate"] - df_sample["control_rate"]
    df_sample["observed_uplift"] = df_sample["observed_uplift"].fillna(0)

    df_sample["percentile"] = (np.arange(1, len(df_sample) + 1) / len(df_sample))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Guardando parquet en {OUTPUT_PATH}")
    df_sample.to_parquet(OUTPUT_PATH, index=False)

    print("Proceso completado.")
    return df_sample


if __name__ == "__main__":
    prepare_processed_data()

