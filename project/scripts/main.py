import databento as db
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def run_analysis():
    # LOADING DATA

    API_KEY = "db-367dAkQvkeFawQ8AH6tEJvrDFE57R"
    client = db.Historical(key=API_KEY)

    end_date = "2025-01-04T00:00:00+00:00"
    start_date = "2025-01-01T00:00:00+00:00"

    dataset = "XNAS.ITCH"
    symbols = ["AAPL", "AMGN", "TSLA", "JPM", "XOM"]
    schema = "mbp-10"

    output_file = "order_book_data.parquet"
    data = client.timeseries.get_range(
        dataset=dataset,
        start=start_date,
        end=end_date,
        symbols=symbols,
        schema=schema,
    )

    data.to_parquet(output_file, compression="snappy")
    # print(f"Data saved to {output_file}")

    df = pd.read_parquet(output_file)
    # print(df.columns)

    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df["ts_event_1s"] = df["ts_event"].dt.floor("1s")
    df_1s = df.groupby(["symbol", "ts_event_1s"], as_index=False).last()
    df_1s = df_1s.drop(columns="ts_event", errors="ignore")
    df_1s = df_1s.rename(columns={"ts_event_1s": "ts_event"})

    df = df_1s

    # OFI CALCULATION

    df = df.sort_values(["symbol", "ts_event"])
    group_cols = [
        "bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00",
        "bid_px_01", "bid_sz_01", "ask_px_01", "ask_sz_01",
        "bid_px_02", "bid_sz_02", "ask_px_02", "ask_sz_02",
        "bid_px_03", "bid_sz_03", "ask_px_03", "ask_sz_03",
        "bid_px_04", "bid_sz_04", "ask_px_04", "ask_sz_04",
    ]

    df_shifted = df.groupby("symbol")[group_cols].shift(1)
    for col in group_cols:
        df[f"prev_{col}"] = df_shifted[col]

    def indicator(condition):
        return np.where(condition, 1, 0)

    for level in range(5):
        df[f"delta_bid_sz_0{level}"] = df[f"bid_sz_0{level}"] - df[f"prev_bid_sz_0{level}"]
        df[f"delta_ask_sz_0{level}"] = df[f"ask_sz_0{level}"] - df[f"prev_ask_sz_0{level}"]

        df[f"delta_bid_px_0{level}"] = df[f"bid_px_0{level}"] - df[f"prev_bid_px_0{level}"]
        df[f"delta_ask_px_0{level}"] = df[f"ask_px_0{level}"] - df[f"prev_ask_px_0{level}"]

        df[f"OFI_level_0{level}"] = (
            df[f"delta_bid_sz_0{level}"] * indicator(df[f"delta_bid_px_0{level}"] >= 0)
            - df[f"delta_ask_sz_0{level}"] * indicator(df[f"delta_ask_px_0{level}"] <= 0)
        )

    ofi_levels = [f"OFI_level_0{level}" for level in range(5)]
    df["OFI_5LEVEL"] = df[ofi_levels].sum(axis=1)

    # PCA

    ofi_cols = [f"OFI_level_0{level}" for level in range(5)]
    df_final = df[["symbol", "ts_event", "OFI_5LEVEL", "bid_px_00", "ask_px_00"] + ofi_cols].copy()

    df_final[ofi_cols] = df_final[ofi_cols].fillna(0)
    ofi_matrix = df_final[ofi_cols].values

    scaler = StandardScaler()
    ofi_matrix_scaled = scaler.fit_transform(ofi_matrix)

    pca = PCA(n_components=1)
    df_final["OFI_5LEVEL_PCA"] = pca.fit_transform(ofi_matrix_scaled)

    float_cols = df_final.select_dtypes(include=[np.float64]).columns
    df_final[float_cols] = df_final[float_cols].astype(np.float32)

    # CROSS IMPACT CODE

    df_final["mid-price"] = (df_final["bid_px_00"] + df_final["ask_px_00"]) / 2
    df_final["returns"] = df_final.groupby("symbol")["mid-price"].pct_change()
    df_final["returns"] = df_final["returns"].fillna(0)
    df_final = df_final.groupby(["ts_event", "symbol"], as_index=False).last()

    all_timestamps = df_final["ts_event"].unique()
    all_symbols = df_final["symbol"].unique()
    df_final = df_final.set_index(["ts_event", "symbol"]).reindex(
        pd.MultiIndex.from_product([all_timestamps, all_symbols], names=["ts_event", "symbol"])
    ).reset_index()

    # print(df_final[df_final["symbol"] == "TSLA"])

    X = df_final.pivot(index="ts_event", columns="symbol", values="OFI_5LEVEL_PCA")
    y = df_final.pivot(index="ts_event", columns="symbol", values="returns")

    cross_impact_coefs = {}

    for symbol in all_symbols:
        temp = pd.concat([X.drop(columns=symbol), y[symbol]], axis=1)
        temp = temp.dropna(how="any", axis=0)
        if temp.shape[0] == 0:
            continue

        alternate_X = temp.drop(columns=symbol).values
        target_Y = temp[symbol].values
        reg = LinearRegression().fit(alternate_X, target_Y)
        print(f"Cross Impact Coefficient for {symbol}: {reg.coef_[0]}")
        cross_impact_coefs[symbol] = reg.coef_[0]

    for lag in [1, 5]:
        shifted = X.shift(lag)
        for symbol in all_symbols:
            df_final[f"OFI_PCA_lag{lag}_{symbol}"] = (
                shifted[symbol]
                .reindex(df_final["ts_event"])
                .values
            )

    lag_cols = [f"OFI_PCA_lag{lag}_{symbol}" for symbol in all_symbols for lag in [1, 5]]
    X_lagged = df_final[lag_cols]

    target_symbol = all_symbols[0]
    temp2 = pd.concat([X_lagged, df_final["returns"].where(df_final["symbol"] == target_symbol)], axis=1)
    temp2 = temp2.fillna(0)

    if temp2.shape[0] == 0:
        print(f"No data available for {target_symbol}")
        return df_final, cross_impact_coefs

    X_features = temp2[lag_cols].values
    Y_target = temp2["returns"].values

    reg2 = LinearRegression().fit(X_features, Y_target)
    y_pred = reg2.predict(X_features)

    mse = mean_squared_error(Y_target, y_pred)
    r2 = r2_score(Y_target, y_pred)
    print(f"MSE: {mse}, R2: {r2}")

    return df_final, cross_impact_coefs

if __name__ == "__main__":
    run_analysis()