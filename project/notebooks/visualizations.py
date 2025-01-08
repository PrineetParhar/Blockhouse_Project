import matplotlib.pyplot as plt
import seaborn as sns
from main import run_analysis

def main():
    df_final, cross_impact_coefs = run_analysis()

    symbols = df_final["symbol"].dropna().unique()
    num_symbols = len(symbols)

    fig, ax = plt.subplots(nrows=num_symbols, figsize=(10,8), sharex=True)

    for i, symbol in enumerate(symbols):
        sym_df = df_final[df_final["symbol"] == symbol]
        ax[i].plot(sym_df["ts_event"], sym_df["OFI_5LEVEL_PCA"])
        ax[i].set_title(f"{symbol} OFI_5LEVEL_PCA Over Time")
        ax[i].set_ylabel("OFI_5LEVEL_PCA")

    ax[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.bar(cross_impact_coefs.keys(), cross_impact_coefs.values(), color="skyblue")
    plt.title("Cross Impact Coefficients (Bar Plot)")
    plt.xlabel("Symbol")
    plt.ylabel("Coefficient")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()