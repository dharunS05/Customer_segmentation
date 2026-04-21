"""
train.py — Feature engineering, KMeans training, model persistence
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

SEGMENT_MAP = {
    0: "Repeat Value Buyers",
    1: "At-Risk Low Value",
    2: "VIP Loyalists",
    3: "Active Growth Customers",
}

SEGMENT_COLORS = {
    "Repeat Value Buyers": "#4A90D9",
    "At-Risk Low Value": "#E05C5C",
    "VIP Loyalists": "#F4A836",
    "Active Growth Customers": "#5CB85C",
}


def engineer_features(rfm: pd.DataFrame):
    """
    Log-transform skewed RFM columns and return (X_scaled array, scaler, feature_df).
    """
    feature_df = pd.DataFrame({
        "recency_log": np.log1p(rfm["recency"]),
        "frequency_log": np.log1p(rfm["frequency"]),
        "monetary_log": np.log1p(rfm["monetary"]),
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    return X_scaled, scaler, feature_df


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """
    Return inertia and silhouette scores for each k in k_range.
    """
    inertia, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    return list(k_range), inertia, sil_scores


def train(rfm: pd.DataFrame, n_clusters: int = 4):
    """
    Engineer features, train KMeans, return (rfm_with_labels, model, scaler).
    """
    X_scaled, scaler, _ = engineer_features(rfm)

    logger.info(f"Training KMeans with n_clusters={n_clusters}")
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = model.fit_predict(X_scaled)
    rfm["segment_name"] = rfm["cluster"].map(SEGMENT_MAP)

    sil = silhouette_score(X_scaled, rfm["cluster"])
    logger.info(f"Silhouette Score: {sil:.4f}")

    return rfm, model, scaler


def save_artifacts(model, scaler, output_dir: str = "models"):
    """Persist model and scaler to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    logger.info(f"Artifacts saved to {output_dir}/")


def load_artifacts(model_dir: str = "models"):
    """Load persisted model and scaler."""
    model = joblib.load(os.path.join(model_dir, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return model, scaler


def profile_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table: one row per cluster with mean RFM + customer count.
    """
    profile = (
        rfm.groupby(["cluster", "segment_name"])
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return profile