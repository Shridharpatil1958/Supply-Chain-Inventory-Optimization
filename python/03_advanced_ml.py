"""
=============================================================
  Retail Analytics – Advanced ML / Forecasting Module
  Author  : Senior Data Analyst
  Version : 1.0.0
  Covers  :
    1. Demand Forecasting    (Moving Average + ARIMA-ready)
    2. Product Segmentation  (RFM-based clustering)
    3. Anomaly Detection     (IQR + Z-score on revenue)
    4. Product Recommender   (co-purchase similarity)
=============================================================
  Install:
    pip install scikit-learn pandas numpy matplotlib
    pip install statsmodels   # for ARIMA
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

CLEANED = "../data/cleaned/"

# ═══════════════════════════════════════════════════════════
#  1. DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════

class DemandForecaster:
    """Rolling-average baseline + ARIMA scaffold."""

    def __init__(self, sales_path: str):
        df = pd.read_csv(sales_path, parse_dates=["sale_date"])
        self.monthly = (
            df.set_index("sale_date")
              .resample("M")["revenue"]
              .sum()
              .rename("revenue")
        )

    def moving_average_forecast(self, window: int = 3, periods: int = 6) -> pd.Series:
        """Simple n-month moving average forecast."""
        ma = self.monthly.rolling(window).mean()
        last_ma = ma.iloc[-1]
        future_idx = pd.date_range(
            self.monthly.index[-1] + pd.offsets.MonthEnd(1),
            periods=periods, freq="M"
        )
        forecast = pd.Series(last_ma, index=future_idx, name="forecast_ma")
        return forecast

    def arima_forecast(self, order=(1,1,1), periods=6):
        """
        ARIMA forecast (requires statsmodels).
        Usage:
            pip install statsmodels
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(self.monthly, order=order).fit()
            return model.forecast(steps=periods)
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(self.monthly, order=order).fit()
            forecast = model.forecast(steps=periods)
            conf_int = model.get_forecast(periods).conf_int()
            return forecast, conf_int
        except ImportError:
            print("statsmodels not installed. Run: pip install statsmodels")
            return self.moving_average_forecast(periods=periods), None

    def plot_forecast(self, save_path: str = None):
        forecast_ma = self.moving_average_forecast()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(self.monthly.index, self.monthly/1e6,
                label="Actual Revenue", color="#4e79a7", lw=2, marker="o", ms=4)
        ax.plot(forecast_ma.index, forecast_ma/1e6,
                label="MA(3) Forecast", color="#f28e2b", lw=2.5,
                ls="--", marker="s", ms=5)
        ax.axvline(self.monthly.index[-1], color="grey", ls=":", alpha=0.6,
                   label="Forecast Start")
        ax.fill_between(forecast_ma.index,
                        forecast_ma/1e6 * 0.9, forecast_ma/1e6 * 1.1,
                        alpha=0.15, color="#f28e2b", label="±10% Confidence Band")
        ax.set_title("Revenue Forecast (Moving Average)", fontsize=15, fontweight="bold")
        ax.set_ylabel("Revenue (₹ Millions)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Forecast plot saved → {save_path}")


# ═══════════════════════════════════════════════════════════
#  2. RFM SEGMENTATION (Customer / Product level)
# ═══════════════════════════════════════════════════════════

class RFMSegmentor:
    """
    Product-level RFM:
      R – Recency     (days since last sale)
      F – Frequency   (number of transactions)
      M – Monetary    (total revenue)
    """

    def __init__(self, sales_path: str, products_path: str):
        sl = pd.read_csv(sales_path, parse_dates=["sale_date"])
        self.p  = pd.read_csv(products_path)
        today   = sl["sale_date"].max()

        self.rfm = sl.groupby("product_id").agg(
            recency   = ("sale_date", lambda x: (today - x.max()).days),
            frequency = ("sale_id",   "count"),
            monetary  = ("revenue",   "sum"),
        ).reset_index()

    def score_and_segment(self, n_clusters: int = 4) -> pd.DataFrame:
        rfm = self.rfm.copy()

        # Standardise
        scaler = StandardScaler()
        X = scaler.fit_transform(rfm[["recency","frequency","monetary"]])

        # KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["segment"] = km.fit_predict(X)

        # Label segments by mean monetary
        seg_means = rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False)
        label_map = {seg: lbl for seg, lbl in
                     zip(seg_means.index,
                         ["Champions","Loyal","At-Risk","Churned"][:n_clusters])}
        rfm["segment_label"] = rfm["segment"].map(label_map)

        # Merge product info
        rfm = rfm.merge(self.p[["product_id","product_name","category"]], on="product_id")
        return rfm

    def silhouette(self, max_k: int = 8) -> dict:
        """Find optimal k via silhouette score."""
        X = StandardScaler().fit_transform(
            self.rfm[["recency","frequency","monetary"]]
        )
        scores = {}
        for k in range(2, max_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            scores[k] = silhouette_score(X, km.labels_)
        return scores


# ═══════════════════════════════════════════════════════════
#  3. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════

class AnomalyDetector:
    """Detect anomalous sales transactions using Z-score + IQR."""

    def __init__(self, sales_path: str):
        self.df = pd.read_csv(sales_path, parse_dates=["sale_date"])

    def detect_zscore(self, col: str = "revenue", threshold: float = 3.0) -> pd.DataFrame:
        df = self.df.copy()
        mean, std = df[col].mean(), df[col].std()
        df["z_score"] = (df[col] - mean) / std
        anomalies = df[df["z_score"].abs() > threshold].copy()
        anomalies["method"] = "Z-Score"
        return anomalies

    def detect_iqr(self, col: str = "revenue") -> pd.DataFrame:
        df = self.df.copy()
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        anomalies = df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)].copy()
        anomalies["method"] = "IQR"
        return anomalies

    def plot_anomalies(self, save_path: str = None):
        anomalies = self.detect_zscore()
        normal    = self.df[~self.df["sale_id"].isin(anomalies["sale_id"])]

        fig, ax = plt.subplots(figsize=(14,5))
        ax.scatter(normal["sale_date"],    normal["revenue"]/1e3,
                   alpha=0.3, s=10, color="#4e79a7", label="Normal")
        ax.scatter(anomalies["sale_date"], anomalies["revenue"]/1e3,
                   alpha=0.85, s=40, color="#e15759", marker="X", label="Anomaly")
        ax.set_title("Revenue Anomaly Detection (Z-Score > 3σ)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Revenue (₹ Thousands)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Anomaly plot saved → {save_path}")
        return len(anomalies), len(anomalies)/len(self.df)*100


# ═══════════════════════════════════════════════════════════
#  4. SIMPLE PRODUCT RECOMMENDER (Co-purchase similarity)
# ═══════════════════════════════════════════════════════════

class ProductRecommender:
    """
    Co-purchase recommender using cosine similarity on
    order co-occurrence matrix.
    """

    def __init__(self, orders_path: str):
        self.orders = pd.read_csv(orders_path)

    def build_similarity_matrix(self) -> pd.DataFrame:
        # Build a product × product co-occurrence count
        # (proxy: products ordered on same date)
        o = self.orders[["order_date","product_id"]].copy()
        o["order_date"] = pd.to_datetime(o["order_date"]).dt.date

        # Products ordered per date
        date_products = o.groupby("order_date")["product_id"].apply(list)

        cooc = {}
        for products in date_products:
            unique = list(set(products))
            for i in range(len(unique)):
                for j in range(i+1, len(unique)):
                    a, b = unique[i], unique[j]
                    key = (min(a,b), max(a,b))
                    cooc[key] = cooc.get(key, 0) + 1

        # Convert to symmetric matrix (top 50 products for speed)
        top_products = (
            self.orders["product_id"].value_counts().head(50).index.tolist()
        )
        mat = pd.DataFrame(0, index=top_products, columns=top_products)
        for (a,b), cnt in cooc.items():
            if a in top_products and b in top_products:
                mat.loc[a,b] += cnt
                mat.loc[b,a] += cnt
        return mat

    def recommend(self, product_id: int, top_n: int = 5) -> list:
        mat = self.build_similarity_matrix()
        if product_id not in mat.index:
            return []
        scores = mat.loc[product_id].sort_values(ascending=False)
        recommendations = scores[scores.index != product_id].head(top_n)
        return recommendations.index.tolist()


# ═══════════════════════════════════════════════════════════
#  MAIN DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("   RETAIL ANALYTICS – ADVANCED ML MODULE")
    print("=" * 60)

    # 1. Forecasting
    print("\n[1] Demand Forecasting")
    forecaster = DemandForecaster(CLEANED + "sales_clean.csv")
    forecast   = forecaster.moving_average_forecast(window=3, periods=6)
    print(f"Next 6-month MA forecast:\n{forecast.round(2)}")
    forecaster.plot_forecast(save_path="../ml_forecast.png")

    # 2. RFM Segmentation
    print("\n[2] RFM Product Segmentation")
    rfm_seg = RFMSegmentor(CLEANED+"sales_clean.csv", CLEANED+"products_clean.csv")
    rfm_df  = rfm_seg.score_and_segment(n_clusters=4)
    print(rfm_df.groupby("segment_label")[["recency","frequency","monetary"]].mean().round(2))

    # 3. Anomaly Detection
    print("\n[3] Anomaly Detection")
    detector = AnomalyDetector(CLEANED + "sales_clean.csv")
    n_anom, pct = detector.plot_anomalies(save_path="../ml_anomalies.png")
    print(f"Detected {n_anom} anomalies ({pct:.2f}% of transactions)")

    # 4. Recommender
    print("\n[4] Product Recommendations for Product_ID=1")
    rec = ProductRecommender(CLEANED + "orders_clean.csv")
    recs = rec.recommend(product_id=1, top_n=5)
    print(f"Recommended product IDs: {recs}")

    print("\n✅  ML Module complete.")