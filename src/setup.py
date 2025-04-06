#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# ---------------------------
# Function: Data Encoding & Cleanup
# ---------------------------
def setup_data(df_results):
    df_encoded = pd.get_dummies(df_results, drop_first=True)
    numeric_cols = df_encoded.select_dtypes(include=np.number).columns
    df_encoded[numeric_cols] = df_encoded[numeric_cols].fillna(df_encoded[numeric_cols].mean())
    print(f"df_encoded ready: shape {df_encoded.shape}")
    return df_encoded

# ---------------------------
# Function: Compute Composite Metrics
# ---------------------------
def compute_composite_metrics(df_results):
    lower = -5
    upper = 5
    def zscore_with_zero(series):
        series = pd.to_numeric(series, errors='coerce')
        z = zscore(series, nan_policy='omit')
        z = np.clip(z, lower, upper)
        return pd.Series(z, index=series.index).fillna(0)
    
    profitability = (zscore_with_zero(df_results["net_profit_margin"]) +
                     zscore_with_zero(df_results["gross_profit_margin"]) +
                     zscore_with_zero(df_results["roa"]) +
                     zscore_with_zero(df_results["roe"]))
    
    liquidity = (zscore_with_zero(df_results["current_ratio"]) +
                 zscore_with_zero(df_results["quick_ratio"]) +
                 zscore_with_zero(df_results["cash_ratio"]))
    
    efficiency = (zscore_with_zero(df_results["inventory_turnover"]) +
                  zscore_with_zero(df_results["accounts_receivable_turnover"]) +
                  zscore_with_zero(df_results["asset_turnover"]))
    
    market_value = (zscore_with_zero(df_results["pe_ratio"]) +
                    zscore_with_zero(df_results["pb_ratio"]) +
                    zscore_with_zero(df_results["dividend_yield"]))
    
    leverage = (zscore_with_zero(df_results["debt_to_equity"]) +
                zscore_with_zero(df_results["debt_ratio"]) +
                zscore_with_zero(df_results["interest_coverage_ratio"]))
    
    scaler = RobustScaler()
    df_results["profitability"] = scaler.fit_transform(profitability.values.reshape(-1, 1))
    df_results["liquidity"]     = scaler.fit_transform(liquidity.values.reshape(-1, 1))
    df_results["efficiency"]    = scaler.fit_transform(efficiency.values.reshape(-1, 1))
    df_results["market_value"]  = scaler.fit_transform(market_value.values.reshape(-1, 1))
    df_results["leverage"]      = scaler.fit_transform(leverage.values.reshape(-1, 1))
    
    return df_results

# ---------------------------
# Function: Perform Principal Component Analysis (PCA)
# ---------------------------
def perform_pca(df_encoded, columns=None, n_components=2):
    if columns is None:
        columns = ["profitability", "liquidity", "efficiency", "market_value", "leverage"]
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_encoded[columns])
    
    pca_model = PCA(n_components=n_components)
    projected = pca_model.fit_transform(scaled_data)
    
    print(f"PCA complete: projected data shape {projected.shape}")
    return projected, pca_model

def compute_clusters(df, projectedData, n_clusters=5, random_state=0, method="kmeans"):
    match (method):
        case "kmeans":
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans = model.fit_predict(projectedData)
            df["cluster"] = kmeans

        case "dendrogram":
            from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
            Z = linkage(projectedData, method='ward')
            # Compute threshold to color the dendrogram for exactly n_clusters segments
            cluster_threshold = Z[-(n_clusters), 2] if n_clusters > 1 else 0
            dendrogram(Z, labels=df["ticker"].tolist(), color_threshold=cluster_threshold)

            # Add matplotlib adjustments: remove y-axis numbers, add x-axis label and title
            import matplotlib.pyplot as plt
            plt.xlabel("Ticker")
            plt.title("Dendrogram of Stocks")
            plt.gca().set_yticklabels([])  # Remove y-axis numbers
            plt.show()

            # Force exactly n_clusters using "maxclust"
            clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
            df["cluster"] = clusters
        case _:
            raise ValueError(f"Invalid method: {method}. Use 'kmeans' or 'dendrogram'.")

    return df



# ---------------------------
# Main execution function
# ---------------------------
def main():
    print("Loading df_results...")
    # Replace with the actual data loading as needed.
    df_results = pd.read_csv("df_results.csv")
    print(f"Data loaded: df_results shape {df_results.shape}")
    
    df_encoded = setup_data(df_results)
    df_results = compute_composite_metrics(df_results)
    
    pca_cols = ["profitability", "liquidity", "efficiency", "market_value", "leverage"]
    projectedData, pca_model = perform_pca(df_encoded, columns=pca_cols, n_components=2)
    
    # Inform the user of key objects available for use.
    print("\nSetup Complete.")
    print("Key Objects Loaded into Memory:")
    print("  df_results      - Main dataframe with composite metrics (shape: {})".format(df_results.shape))
    print("  df_encoded      - Encoded dataframe (shape: {})".format(df_encoded.shape))
    print("  projectedData   - PCA projected data (shape: {})".format(projectedData.shape))
    print("  pca_model       - Fitted PCA model (Explained variance ratio: {})".format(pca_model.explained_variance_ratio_))

if __name__ == "__main__":
    main()