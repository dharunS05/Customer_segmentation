"""
preprocess.py — Data cleaning and RFM feature engineering
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw Online Retail II CSV and return a cleaned DataFrame.
    Steps: standardize columns, drop duplicates, remove nulls,
    filter bad transactions, parse dates, compute revenue.
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove missing customer_id and invoicedate
    df = df.dropna(subset=["customer_id", "invoicedate"])

    # Remove cancelled invoices (start with 'C')
    df = df[~df["invoice"].astype(str).str.startswith("C")]

    # Remove bad quantity / price
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]

    # Type conversions
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # Revenue
    df["revenue"] = df["quantity"] * df["price"]

    logger.info(f"Cleaned shape: {df.shape} | Unique customers: {df['customer_id'].nunique()}")
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features per customer.
    """
    snapshot_date = df["invoicedate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("invoicedate", lambda x: (snapshot_date - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    logger.info(f"RFM built — shape: {rfm.shape}")
    return rfm