"""
pipeline.py — End-to-end CLI runner: clean → RFM → train → save
Run: python pipeline.py --input data/online_retail_II.csv
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_and_clean, build_rfm
from train import train, save_artifacts, profile_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(input_path: str, output_dir: str = "data", model_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load & clean
    df = load_and_clean(input_path)
    df.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
    logger.info("Saved cleaned_data.csv")

    # 2. RFM
    rfm = build_rfm(df)
    rfm.to_csv(os.path.join(output_dir, "rfm.csv"), index=False)
    logger.info("Saved rfm.csv")

    # 3. Train
    rfm_labeled, model, scaler = train(rfm, n_clusters=4)
    rfm_labeled.to_csv(os.path.join(output_dir, "final_customer_segments.csv"), index=False)
    logger.info("Saved final_customer_segments.csv")

    # 4. Profile
    profile = profile_clusters(rfm_labeled)
    profile.to_csv(os.path.join(output_dir, "cluster_profile.csv"), index=False)
    logger.info("Saved cluster_profile.csv")

    # 5. Save artifacts
    save_artifacts(model, scaler, model_dir)

    logger.info("✅ Pipeline complete!")
    print("\nCluster Profile:")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Segmentation Pipeline")
    parser.add_argument("--input", required=True, help="Path to raw Online Retail II CSV")
    parser.add_argument("--output_dir", default="data", help="Directory for output CSVs")
    parser.add_argument("--model_dir", default="models", help="Directory for model artifacts")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, args.model_dir)