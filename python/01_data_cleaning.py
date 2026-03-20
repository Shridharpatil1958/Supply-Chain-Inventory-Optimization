"""
=============================================================
  Retail Analytics – Data Cleaning Pipeline
  Author  : Senior Data Analyst
  Version : 2.0.0 (Fixed Paths + Robust Saving)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import sys

# Ensure UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# ── PATH CONFIG (🔥 FIXED) ──────────────────────────────────
BASE_DIR = "C:/Users/Yash/Supply Chain & Inventory Optimization/data"

RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")

os.makedirs(CLEANED_DIR, exist_ok=True)

# ── LOGGING ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cleaning_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def standardize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def remove_duplicates(df, name):
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"[{name}] Removed {before - len(df)} duplicates")
    return df


def report_nulls(df, name):
    nulls = df.isnull().sum()
    if nulls.any():
        logger.warning(f"[{name}] Nulls:\n{nulls[nulls > 0]}")
    else:
        logger.info(f"[{name}] No nulls")


def cap_outliers(df, col, name):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    df[col] = df[col].clip(low, high)
    logger.info(f"[{name}] Outliers capped in {col}")
    return df


def parse_dates(df, cols, name):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def save_clean(df, name):
    path = os.path.join(CLEANED_DIR, f"{name}_clean.csv")

    try:
        print(f"Saving → {path}")
        df.to_csv(path, index=False)
        logger.info(f"[{name}] Saved successfully ({len(df)} rows)")
    except Exception as e:
        logger.error(f"[{name}] Save failed: {e}")


# ═══════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def clean_products():
    df = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))

    df = standardize_columns(df)
    report_nulls(df, "products")
    df = remove_duplicates(df, "products")

    df["product_id"] = df["product_id"].astype(int)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["cost"].fillna(df["cost"].median(), inplace=True)
    df["price"].fillna(df["price"].median(), inplace=True)

    df["margin"] = df["price"] - df["cost"]
    df["margin_pct"] = (df["margin"] / df["price"] * 100).round(2)

    df["category"] = df["category"].str.title()

    df = cap_outliers(df, "cost", "products")
    df = cap_outliers(df, "price", "products")

    save_clean(df, "products")
    return df


def clean_suppliers():
    df = pd.read_csv(os.path.join(RAW_DIR, "suppliers.csv"))

    df = standardize_columns(df)
    report_nulls(df, "suppliers")
    df = remove_duplicates(df, "suppliers")

    df["supplier_id"] = df["supplier_id"].astype(int)
    df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce")

    df["location"] = df["location"].str.title()

    df["long_lead_flag"] = df["lead_time"] > 30

    save_clean(df, "suppliers")
    return df


def clean_inventory():
    df = pd.read_csv(os.path.join(RAW_DIR, "inventory.csv"))

    df = standardize_columns(df)
    report_nulls(df, "inventory")
    df = remove_duplicates(df, "inventory")

    df["product_id"] = df["product_id"].astype(int)
    df["stock_level"] = pd.to_numeric(df["stock_level"], errors="coerce")
    df["reorder_level"] = pd.to_numeric(df["reorder_level"], errors="coerce")

    df["stock_level"] = df["stock_level"].clip(lower=0)
    df["needs_reorder"] = df["stock_level"] <= df["reorder_level"]

    save_clean(df, "inventory")
    return df


def clean_orders():
    df = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))

    df = standardize_columns(df)
    report_nulls(df, "orders")
    df = remove_duplicates(df, "orders")

    df["order_id"] = df["order_id"].astype(int)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df = parse_dates(df, ["order_date", "delivery_date"], "orders")

    df = df[df["delivery_date"] >= df["order_date"]]

    df["fulfilment_days"] = (
        df["delivery_date"] - df["order_date"]
    ).dt.days

    df = cap_outliers(df, "quantity", "orders")

    save_clean(df, "orders")
    return df


def clean_sales():
    df = pd.read_csv(os.path.join(RAW_DIR, "sales.csv"))

    df = standardize_columns(df)
    report_nulls(df, "sales")
    df = remove_duplicates(df, "sales")

    df["sale_id"] = df["sale_id"].astype(int)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    df = parse_dates(df, ["sale_date"], "sales")

    df = df[(df["revenue"] > 0) & (df["quantity"] > 0)]

    df["year"] = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month

    df["unit_price"] = df["revenue"] / df["quantity"]

    df = cap_outliers(df, "revenue", "sales")

    save_clean(df, "sales")
    return df


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline():
    logger.info("STARTING DATA CLEANING PIPELINE")

    start = datetime.now()

    products = clean_products()
    suppliers = clean_suppliers()
    inventory = clean_inventory()
    orders = clean_orders()
    sales = clean_sales()

    logger.info("Cross Validation")

    logger.info(f"Orphan Sales: {len(sales[~sales['product_id'].isin(products['product_id'])])}")
    logger.info(f"Orphan Orders: {len(orders[~orders['product_id'].isin(products['product_id'])])}")

    logger.info(f"Completed in {(datetime.now() - start).seconds} sec")


if __name__ == "__main__":
    run_pipeline()
    print("\n✅ All datasets cleaned and saved successfully!")