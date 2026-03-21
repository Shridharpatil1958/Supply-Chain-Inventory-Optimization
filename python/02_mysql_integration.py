"""
=============================================================
  Retail Analytics – Python → MySQL Integration
  Author  : Senior Data Analyst
  Version : 2.0.0
  Fixes   :
    v1 → v2:
      - Use 127.0.0.1 instead of localhost (DNS resolution fix)
      - Encode special chars in password URL (%40 for @)
      - Force InnoDB on all tables (fixes table-full / MyISAM error)
      - Connection test before upload
      - safe_upload() helper for all fact + dim tables
      - Graceful error handling with clear messages
=============================================================
  Prerequisites:
    pip install mysql-connector-python sqlalchemy pandas
=============================================================
"""

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
import logging
import os
from urllib.parse import quote_plus
from datetime import datetime, date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────
# Either set environment variables, or edit the defaults below.
# NOTE: Use 127.0.0.1, NOT "localhost" — avoids DNS resolution
#       failure on Windows with mysql-connector-python.

DB_HOST     = os.getenv("DB_HOST",     "127.0.0.1")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_USER     = os.getenv("DB_USER",     "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root@2026")   # supports special chars
DB_NAME     = os.getenv("DB_NAME",     "supply_chain_analytics")

CLEANED_DIR = "../data/cleaned/"


# ═══════════════════════════════════════════════════════════
#  CONNECTION HELPERS
# ═══════════════════════════════════════════════════════════

def get_engine():
    """
    Return a SQLAlchemy engine.
    - Uses 127.0.0.1 (not localhost) to avoid DNS issues on Windows.
    - URL-encodes the password so special characters like @ don't
      break the connection string parser.
    """
    password_encoded = quote_plus(DB_PASSWORD)   # root@2026 → root%402026
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{password_encoded}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?charset=utf8mb4"
    )
    return create_engine(url, echo=False)


def test_connection(engine) -> bool:
    """
    Verify the connection works before attempting any uploads.
    Returns True on success, False on failure.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅  MySQL connection successful — host=%s  db=%s", DB_HOST, DB_NAME)
        return True
    except Exception as e:
        logger.error("❌  Cannot connect to MySQL.")
        logger.error("    Error  : %s", str(e))
        logger.error("    Check  : 1) MySQL service is running (services.msc → MySQL80)")
        logger.error("             2) Database '%s' exists", DB_NAME)
        logger.error("             3) Host / port / credentials are correct")
        logger.error("    Fix    : CREATE DATABASE IF NOT EXISTS %s;", DB_NAME)
        return False


# ═══════════════════════════════════════════════════════════
#  SAFE UPLOAD HELPER  (fixes MyISAM table-full error)
# ═══════════════════════════════════════════════════════════

def safe_upload(df: pd.DataFrame, table_name: str, engine,
                chunksize: int = 500) -> None:
    """
    Upload a DataFrame to MySQL with InnoDB engine forced.

    Why this is needed:
      pandas to_sql with if_exists='replace' can create tables using
      MySQL's default storage engine. On some Windows installs that
      defaults to MyISAM, which has a ~4 GB per-table file limit.
      With 10,000+ rows of wide data this limit can be hit, producing:
          1114 (HY000): The table '...' is full

    This function:
      1. Drops the table if it already exists.
      2. Lets pandas CREATE + INSERT the data (if_exists='replace').
      3. Immediately ALTERs the engine to InnoDB, which has no such limit.
    """
    logger.info("  Uploading  %-20s  (%s rows) ...", table_name, f"{len(df):,}")
    try:
        # Step 1 - drop cleanly so pandas can recreate
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))

        # Step 2 - pandas creates + fills the table
        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
            chunksize=chunksize,
        )

        # Step 3 - force InnoDB (eliminates MyISAM 4 GB limit)
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE `{table_name}` ENGINE=InnoDB"))

        logger.info("  OK  %-20s  uploaded successfully.", table_name)

    except Exception as e:
        logger.error("  FAIL  Failed to upload table '%s': %s", table_name, str(e))
        raise


# ═══════════════════════════════════════════════════════════
#  DATE DIMENSION GENERATOR
# ═══════════════════════════════════════════════════════════

def build_date_dimension(start: str = "2023-01-01",
                         end:   str = "2025-12-31") -> pd.DataFrame:
    """Generate a full date spine for time-intelligence queries."""
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"full_date": dates})
    df["date_key"]     = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["year"]         = df["full_date"].dt.year
    df["quarter"]      = df["full_date"].dt.quarter
    df["month"]        = df["full_date"].dt.month
    df["month_name"]   = df["full_date"].dt.strftime("%B")
    df["week_of_year"] = df["full_date"].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df["full_date"].dt.day
    df["day_name"]     = df["full_date"].dt.strftime("%A")
    df["is_weekend"]   = df["full_date"].dt.dayofweek >= 5
    return df


# ═══════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════

def load_all() -> dict:
    """Load all cleaned CSVs into DataFrames."""
    logger.info("Loading cleaned datasets from '%s' ...", CLEANED_DIR)
    datasets = {
        "products" : pd.read_csv(CLEANED_DIR + "products_clean.csv"),
        "suppliers": pd.read_csv(CLEANED_DIR + "suppliers_clean.csv"),
        "inventory": pd.read_csv(CLEANED_DIR + "inventory_clean.csv"),
        "orders"   : pd.read_csv(
            CLEANED_DIR + "orders_clean.csv",
            parse_dates=["order_date", "delivery_date"]
        ),
        "sales"    : pd.read_csv(
            CLEANED_DIR + "sales_clean.csv",
            parse_dates=["sale_date"]
        ),
    }
    for name, df in datasets.items():
        logger.info("  Loaded %-12s  shape=%s", name, df.shape)
    return datasets


# ═══════════════════════════════════════════════════════════
#  UPLOAD PIPELINE
# ═══════════════════════════════════════════════════════════

def upload_all(engine, datasets: dict) -> None:
    """
    Upload all dimension and fact tables to MySQL.
    Load order: dimensions first, then facts (FK safety).
    """
    logger.info("=" * 55)
    logger.info("  UPLOAD PIPELINE START")
    logger.info("=" * 55)

    # ── dim_supplier ──────────────────────────────────────
    sup = datasets["suppliers"][
        ["supplier_id", "supplier_name", "lead_time", "location"]
    ].copy()
    safe_upload(sup, "dim_supplier", engine)

    # ── dim_product ───────────────────────────────────────
    prod = datasets["products"][
        ["product_id", "product_name", "category", "cost", "price"]
    ].copy()
    safe_upload(prod, "dim_product", engine)

    # ── dim_date ──────────────────────────────────────────
    dim_date = build_date_dimension()
    safe_upload(dim_date, "dim_date", engine, chunksize=1000)

    # ── dim_warehouse ─────────────────────────────────────
    warehouses = pd.DataFrame({
        "warehouse_name": datasets["inventory"]["warehouse"].unique()
    })
    warehouses["warehouse_id"] = range(1, len(warehouses) + 1)
    safe_upload(warehouses, "dim_warehouse", engine)

    # ── fact_inventory ────────────────────────────────────
    inv = datasets["inventory"].merge(
        warehouses,
        left_on="warehouse",
        right_on="warehouse_name"
    )[["product_id", "warehouse_id", "stock_level", "reorder_level"]].copy()
    inv["needs_reorder"] = inv["stock_level"] <= inv["reorder_level"]
    safe_upload(inv, "fact_inventory", engine)

    # ── fact_orders ───────────────────────────────────────
    ord_df = datasets["orders"][
        ["order_id", "product_id", "order_date", "delivery_date", "quantity"]
    ].copy()
    ord_df["order_date"]    = ord_df["order_date"].dt.date
    ord_df["delivery_date"] = ord_df["delivery_date"].dt.date
    safe_upload(ord_df, "fact_orders", engine, chunksize=1000)

    # ── fact_sales ────────────────────────────────────────
    sal = datasets["sales"][
        ["sale_id", "product_id", "sale_date", "quantity", "revenue"]
    ].copy()
    sal["sale_date"] = sal["sale_date"].dt.date
    safe_upload(sal, "fact_sales", engine, chunksize=1000)

    logger.info("=" * 55)
    logger.info("  ALL TABLES UPLOADED SUCCESSFULLY")
    logger.info("=" * 55)


# ═══════════════════════════════════════════════════════════
#  ANALYTICAL QUERIES
# ═══════════════════════════════════════════════════════════

QUERIES = {

    "total_kpis": """
        SELECT
            COUNT(DISTINCT s.sale_id)   AS total_transactions,
            SUM(s.quantity)             AS total_units_sold,
            ROUND(SUM(s.revenue), 2)    AS total_revenue,
            ROUND(AVG(s.revenue), 2)    AS avg_order_value
        FROM fact_sales s
    """,

    "monthly_revenue": """
        SELECT
            d.year,
            d.month,
            d.month_name,
            ROUND(SUM(s.revenue), 2)  AS monthly_revenue,
            SUM(s.quantity)           AS units_sold
        FROM fact_sales s
        JOIN dim_date d ON s.sale_date = d.full_date
        GROUP BY d.year, d.month, d.month_name
        ORDER BY d.year, d.month
    """,

    "top10_products": """
        SELECT
            p.product_id,
            p.product_name,
            p.category,
            SUM(s.quantity)           AS units_sold,
            ROUND(SUM(s.revenue), 2)  AS total_revenue
        FROM fact_sales s
        JOIN dim_product p ON s.product_id = p.product_id
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY total_revenue DESC
        LIMIT 10
    """,

    "category_performance": """
        SELECT
            p.category,
            COUNT(DISTINCT p.product_id)  AS product_count,
            SUM(s.quantity)               AS units_sold,
            ROUND(SUM(s.revenue), 2)      AS total_revenue,
            ROUND(AVG(s.revenue), 2)      AS avg_sale_value
        FROM fact_sales s
        JOIN dim_product p ON s.product_id = p.product_id
        GROUP BY p.category
        ORDER BY total_revenue DESC
    """,

    "inventory_alerts": """
        SELECT
            p.product_name,
            p.category,
            w.warehouse_name,
            i.stock_level,
            i.reorder_level,
            (i.reorder_level - i.stock_level) AS deficit
        FROM fact_inventory i
        JOIN dim_product   p ON i.product_id   = p.product_id
        JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
        WHERE i.stock_level <= i.reorder_level
        ORDER BY deficit DESC
        LIMIT 20
    """,

    "yearly_comparison": """
        SELECT
            d.year,
            d.quarter,
            ROUND(SUM(s.revenue), 2)  AS quarterly_revenue,
            SUM(s.quantity)           AS units_sold
        FROM fact_sales s
        JOIN dim_date d ON s.sale_date = d.full_date
        GROUP BY d.year, d.quarter
        ORDER BY d.year, d.quarter
    """,
}


def run_queries(engine) -> dict:
    """Execute all analytical queries and print results."""
    logger.info("=" * 55)
    logger.info("  RUNNING ANALYTICAL QUERIES")
    logger.info("=" * 55)
    results = {}
    for name, sql in QUERIES.items():
        try:
            logger.info("Running: %s", name)
            df = pd.read_sql(text(sql), engine)
            results[name] = df
            print(f"\n{'─' * 55}")
            print(f"  {name.upper().replace('_', ' ')}")
            print(f"{'─' * 55}")
            print(df.to_string(index=False))
        except Exception as e:
            logger.warning("Query '%s' failed: %s", name, str(e))
    return results


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    logger.info("=" * 55)
    logger.info("  RETAIL ANALYTICS — MySQL Integration v2.0")
    logger.info("=" * 55)

    # Step 1 — Build engine
    logger.info("Building SQLAlchemy engine ...")
    engine = get_engine()

    # Step 2 — Test connection BEFORE doing anything else
    if not test_connection(engine):
        raise SystemExit(
            "\n  Aborting. Fix the MySQL connection first.\n"
            "  Common fixes:\n"
            "  1) Start MySQL service  ->  net start mysql80\n"
            "  2) Create database      ->  CREATE DATABASE supply_chain_analytics;\n"
            "  3) Check password in DB_PASSWORD at top of this file\n"
        )

    # Step 3 — Load cleaned CSVs
    datasets = load_all()

    # Step 4 — Upload to MySQL
    upload_all(engine, datasets)

    # Step 5 — Run analytical queries
    results = run_queries(engine)

    logger.info("Pipeline complete.")