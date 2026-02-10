import numpy as np

def precompute_uplift_stats(df):
    """
    Precomputa estadísticas causales independientes del coste y del revenue.
    Se ejecuta UNA sola vez al iniciar la app.
    """
    percentiles=np.linspace(0.01, 1.0, 100)

    N = len(df)

    # Baseline: conversión media del control (no treatment)
    control_mean_conv = df[df["treatment"] == 0]["conversion"].mean()
    print(control_mean_conv)

    # Uplift global (random targeting)
    global_treat_rate = df[df["treatment"] == 1]["conversion"].mean()
    global_uplift = global_treat_rate - control_mean_conv

    uplift_local = []

    for p in percentiles:
        k = int(p * N)
        subset = df.iloc[:k]

        treat_rate = subset[subset["treatment"] == 1]["conversion"].mean()
        control_rate = subset[subset["treatment"] == 0]["conversion"].mean()
        uplift_local.append(treat_rate - control_rate)

    return {
        "percentiles": percentiles,
        "uplift_local": np.array(uplift_local),
        "uplift_global": global_uplift,
        "baseline": control_mean_conv
    }



# --------------------------------------------------
# 2) Cálculo de negocio (barato, depende de inputs)
# --------------------------------------------------

def compute_profit_curves(
    uplift_stats,
    population_size,
    cpm,
    impressions_per_user,
    conversion_value
):
    """
    Cálculo instantáneo de beneficios incrementales.
    """

    percentiles = uplift_stats["percentiles"]
    uplift_local = uplift_stats["uplift_local"]
    uplift_global = uplift_stats["uplift_global"]

    cost_per_user = cpm * impressions_per_user / 1000
    k = percentiles * population_size

    profit_uplift = k * (uplift_local * conversion_value - cost_per_user)
    profit_random = k * (uplift_global * conversion_value - cost_per_user)

    best_idx = np.argmax(profit_uplift)

    return {
        "percentiles": percentiles,
        "profit_uplift": profit_uplift,
        "profit_random": profit_random,
        "best_percentile": percentiles[best_idx],
        "best_profit": profit_uplift[best_idx],
        "random_profit_at_best": profit_random[best_idx],
    }
