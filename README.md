# SupplySight — Supply Chain & Inventory Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?logo=mysql&logoColor=white)
![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A complete, industry-grade supply chain analytics solution — from raw CSV data through Python cleaning, MySQL storage, SQL analysis, to a fully interactive 4-page Power BI dashboard with forecasting and conditional formatting.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Metrics](#key-metrics)
3. [Technology Stack](#technology-stack)
4. [Repository Structure](#repository-structure)
5. [Step 1 — Data Cleaning Python](#step-1--data-cleaning-python)
6. [Step 2 — MySQL Database](#step-2--mysql-database)
7. [Step 3 — SQL Query Library](#step-3--sql-query-library)
8. [Step 4 — Power BI Dashboard](#step-4--power-bi-dashboard)
9. [Step 5 — DAX Measures](#step-5--dax-measures)
10. [Step 6 — Conditional Formatting](#step-6--conditional-formatting)
11. [Step 7 — Business Insights](#step-7--business-insights)
12. [Setup and Installation](#setup-and-installation)

---

## Project Overview

SupplySight analyses 10,000+ sales transactions, 8,000+ orders, 500 products, 50 suppliers and 500 inventory records across 5 Indian warehouses. The goal is to deliver actionable intelligence across four business domains: Sales Performance, Inventory Management, Supplier Efficiency, and Revenue Forecasting.

**Full pipeline:**

```
Raw CSVs
   |
   v
Python Cleaning (Pandas)
   |
   v
Cleaned CSVs  ──>  MySQL Star Schema  ──>  Power BI Dashboard (4 pages)
                        |
                        v
                   SQL Analytics (31 queries)
                   ML Forecasting (MA3 + ARIMA)
```

---

## Key Metrics

| Metric | Value | Metric | Value |
|---|---|---|---|
| Total Revenue | Rs 102.5M | Sales Transactions | 10,000 |
| Orders Processed | 8,000 | Products (SKUs) | 500 |
| Suppliers | 50 | Warehouses | 5 |
| Avg Gross Margin | 33.07% | Reorder Alerts | 78 SKUs |
| Avg Fulfilment Time | 8.55 days | Top Category | Accessories |
| Best Product | Product_139 | Peak Day | Sunday |

---

## Technology Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data Cleaning | Python 3.8 / Pandas | Standardise, validate, engineer features |
| Database | MySQL 8.0 | Star schema storage and analytical queries |
| BI Dashboard | Power BI Desktop | 4-page interactive dashboard |
| Forecasting | Power BI ETS + Python MA(3) | 6-month revenue forecast |
| ML Features | scikit-learn | RFM segmentation, anomaly detection |
| Version Control | Git / GitHub | Full project versioning |

---

## Repository Structure

```
supplysight/
|
+-- data/
|   +-- raw/                        # Original CSV files (do not modify)
|   |   +-- products.csv
|   |   +-- suppliers.csv
|   |   +-- inventory.csv
|   |   +-- orders.csv
|   |   +-- sales.csv
|   |
|   +-- cleaned/                    # Cleaned and enriched CSVs
|       +-- products_clean.csv
|       +-- suppliers_clean.csv
|       +-- inventory_clean.csv
|       +-- orders_clean.csv
|       +-- sales_clean.csv
|
+-- python/
|   +-- 01_data_cleaning.py         # Production cleaning pipeline
|   +-- 02_mysql_integration.py     # Python to MySQL upload with InnoDB fix
|
+-- sql/
|   +-- analytics_queries.sql       # 31 queries + 3 stored views
|
+-- ml/
|   +-- 04_advanced_ml.py           # Forecasting, RFM, anomaly detection
|
+-- powerbi/
|   +-- SupplySight.pbix            # Power BI report file
|
+-- docs/
|   +-- dashboard_executive_overview.png
|   +-- dashboard_sales_analysis.png
|   +-- dashboard_inventory_analysis.png
|   +-- dashboard_supplier_analysis.png
|
+-- requirements.txt
+-- README.md
```

---

## Step 1 — Data Cleaning Python

**File:** `python/01_data_cleaning.py`

### Datasets

| File | Rows | Columns | Role |
|---|---|---|---|
| products.csv | 500 | 5 | Dimension — product catalogue |
| suppliers.csv | 50 | 4 | Dimension — supplier master |
| inventory.csv | 500 | 4 | Snapshot Fact — stock levels per warehouse |
| orders.csv | 8,000 | 5 | Fact — purchase orders with delivery dates |
| sales.csv | 10,000 | 5 | Fact — sales transactions with revenue |

### Cleaning Steps Applied

1. **Column standardisation** — all names converted to snake_case via regex
2. **Duplicate removal** — `drop_duplicates()` with before/after logging
3. **Date parsing** — `pd.to_datetime(errors='coerce')` with NaT detection. Cross-validates delivery_date >= order_date
4. **Type enforcement** — product_id to int, cost/price to float64, dates to datetime64
5. **Outlier capping** — IQR winsorisation on revenue, quantity, cost, price. Caps not drops
6. **Negative stock correction** — stock_level clipped to minimum 0
7. **Cross-table referential integrity** — zero orphaned product_id records found

### Derived Columns Added

| Column | Table | Formula |
|---|---|---|
| margin | products | price minus cost |
| margin_pct | products | (margin / price) x 100 |
| needs_reorder | inventory | stock_level <= reorder_level |
| fulfilment_days | orders | delivery_date minus order_date in days |
| year / month / quarter / week | sales | extracted from sale_date |
| unit_price_realised | sales | revenue / quantity |

### Key Code Snippets

```python
# Outlier capping using IQR winsorisation (cap not drop)
def cap_outliers_iqr(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


# Date parsing with error handling
def parse_dates(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# Column name standardisation to snake_case
def standardize_columns(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[\s\-]+', '_', regex=True)
    )
    return df


# Cross-table validation
orphan_sales = sales[~sales['product_id'].isin(products['product_id'])]
print(f'Orphan sales rows: {len(orphan_sales)}')   # Expected: 0
```

### Run the Pipeline

```bash
python python/01_data_cleaning.py
```

Output: 5 cleaned CSVs saved to `data/cleaned/`

---

## Step 2 — MySQL Database

**File:** `python/02_mysql_integration.py`

### Star Schema Design

```
                 dim_date (1,096 rows)
                      |
                      |
dim_product -----  fact_sales (10,000 rows)
(500 rows)   |
             |
             +-----  fact_orders (8,000 rows)
             |
             +-----  fact_inventory (500 rows) ----- dim_warehouse (5 rows)

dim_supplier (50 rows)
```

### Tables

| Table | Type | Rows | Primary Key |
|---|---|---|---|
| dim_product | Dimension | 500 | product_id |
| dim_supplier | Dimension | 50 | supplier_id |
| dim_warehouse | Dimension | 5 | warehouse_id |
| dim_date | Dimension | 1,096 | date_key |
| fact_sales | Fact | 10,000 | sale_id |
| fact_orders | Fact | 8,000 | order_id |
| fact_inventory | Snapshot Fact | 500 | auto_increment |

### Connection Setup

```python
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# Use 127.0.0.1 not localhost — avoids DNS resolution error on Windows
# URL-encode the password — @ in password must become %40
password_encoded = quote_plus("root@2026")   # becomes root%402026

url = (
    f"mysql+mysqlconnector://root:{password_encoded}"
    f"@127.0.0.1:3306/supply_chain_analytics"
    f"?charset=utf8mb4"
)
engine = create_engine(url, echo=False)
```

### InnoDB Fix — Prevents Table-Full Error 1114

Without this fix you get: `1114 (HY000): The table 'fact_sales' is full`

pandas `to_sql` sometimes creates tables as MyISAM which has a 4GB limit. Fix:

```python
def safe_upload(df, table_name, engine, chunksize=500):
    # Step 1 — drop existing table
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))

    # Step 2 — pandas creates and fills the table
    df.to_sql(table_name, engine, if_exists="replace",
              index=False, chunksize=chunksize)

    # Step 3 — force InnoDB (removes MyISAM 4GB limit)
    with engine.connect() as conn:
        conn.execute(text(f"ALTER TABLE `{table_name}` ENGINE=InnoDB"))

    print(f"Uploaded {table_name} ({len(df):,} rows)")
```

### Run the Integration

```bash
# First create the database in MySQL Workbench
CREATE DATABASE IF NOT EXISTS supply_chain_analytics;

# Then run the Python script
python python/02_mysql_integration.py
```

Expected output:

```
[INFO] MySQL connection successful — host=127.0.0.1  db=supply_chain_analytics
[INFO] Uploading dim_supplier     (50 rows)
[INFO] Uploading dim_product      (500 rows)
[INFO] Uploading dim_date         (1096 rows)
[INFO] Uploading dim_warehouse    (5 rows)
[INFO] Uploading fact_inventory   (500 rows)
[INFO] Uploading fact_orders      (8000 rows)
[INFO] Uploading fact_sales       (10000 rows)
[INFO] ALL TABLES UPLOADED SUCCESSFULLY
```

---

## Step 3 — SQL Query Library

**File:** `sql/analytics_queries.sql`

737 lines | 31 queries | 3 stored views

### Query Sections

| Section | Queries | What It Covers |
|---|---|---|
| 1. DB Overview | 3 | Table counts, row verification, previews |
| 2. Executive KPIs | 3 | Total revenue, YoY comparison, growth % |
| 3. Sales Analysis | 8 | Monthly trend, quarterly, category, top 10, bottom 10, day of week |
| 4. Inventory | 5 | Warehouse overview, reorder alerts, turnover ratio, overstocked SKUs |
| 5. Supplier | 4 | Location performance, fulfilment stats, delivery buckets |
| 6. Product | 3 | Full scorecard, margin analysis, dead stock detection |
| 7. Advanced | 5 | ABC segmentation, MoM + MA3, cross-analysis, quarterly pivot |
| 8. Stored Views | 3 | v_sales_enriched, v_inventory_alerts, v_supplier_performance |

### Sample Queries

```sql
-- Total Revenue KPIs
-- Result: revenue=102,503,482 | transactions=10,000 | avg_order=10,250
SELECT
    COUNT(DISTINCT s.sale_id)                     AS total_transactions,
    SUM(s.quantity)                               AS total_units_sold,
    ROUND(SUM(s.revenue), 2)                      AS total_revenue,
    ROUND(AVG(s.revenue), 2)                      AS avg_order_value,
    ROUND(SUM(s.revenue - s.quantity*p.cost), 2)  AS gross_profit
FROM fact_sales s
JOIN dim_product p ON s.product_id = p.product_id;


-- Top 10 Products by Revenue
-- Result: Product_139=605,886 | Product_173=569,190 | Product_384=552,066
SELECT
    p.product_name,
    p.category,
    SUM(s.quantity)            AS units_sold,
    ROUND(SUM(s.revenue), 2)   AS total_revenue,
    RANK() OVER (ORDER BY SUM(s.revenue) DESC) AS revenue_rank
FROM fact_sales s
JOIN dim_product p ON s.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC
LIMIT 10;


-- Reorder Alerts by Warehouse
-- Result: Bangalore=18 | Mumbai=17 | Hyderabad=16 | Chennai=14 | Delhi=13
SELECT
    w.warehouse_name,
    SUM(CASE WHEN i.needs_reorder = 1 THEN 1 ELSE 0 END) AS alert_count,
    ROUND(AVG(i.stock_level) / AVG(i.reorder_level), 2)  AS coverage_ratio
FROM fact_inventory i
JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
GROUP BY w.warehouse_name
ORDER BY alert_count DESC;


-- ABC Revenue Segmentation using CTE
WITH product_revenue AS (
    SELECT p.product_id, p.product_name, p.category,
           ROUND(SUM(s.revenue), 2) AS total_revenue
    FROM fact_sales s
    JOIN dim_product p ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name, p.category
),
revenue_cumulative AS (
    SELECT *,
        ROUND(SUM(total_revenue) OVER (
            ORDER BY total_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / SUM(total_revenue) OVER () * 100, 2) AS cumulative_pct
    FROM product_revenue
)
SELECT product_name, category, total_revenue, cumulative_pct,
    CASE
        WHEN cumulative_pct <= 80 THEN 'A — Star'
        WHEN cumulative_pct <= 95 THEN 'B — Core'
        ELSE                           'C — Long Tail'
    END AS abc_segment
FROM revenue_cumulative
ORDER BY total_revenue DESC;
```

### How to Run in MySQL Workbench

1. Open MySQL Workbench and connect to `127.0.0.1:3306`
2. File → Open SQL Script → select `analytics_queries.sql`
3. Run `USE supply_chain_analytics;` first
4. Select any query block and press `Ctrl + Shift + Enter` to run just that block
5. The 3 views in Section 8 only need to be created once

---

## Step 4 — Power BI Dashboard

**File:** `powerbi/SupplySight.pbix`

### How to Connect Power BI to MySQL

1. Open Power BI Desktop
2. Home → Get Data → MySQL Database
3. Server: `127.0.0.1`, Database: `supply_chain_analytics`
4. Import tables: `fact_sales`, `fact_orders`, `fact_inventory`, `dim_product`, `dim_supplier`, `dim_warehouse`, `dim_date`

### Relationships to Set in Model View

```
fact_sales.product_id      → dim_product.product_id     (Many to One)
fact_sales.sale_date       → dim_date.full_date          (Many to One)
fact_orders.product_id     → dim_product.product_id     (Many to One)
fact_inventory.product_id  → dim_product.product_id     (Many to One)
fact_inventory.warehouse_id → dim_warehouse.warehouse_id (Many to One)
```

---

### Page 1 — Executive Overview

![Executive Overview]
(<img width="1080" height="614" alt="image" src="https://github.com/user-attachments/assets/5426f529-61a5-4dca-879b-205affe5c87e" />)

**KPI Cards:**
- Total Revenue: Rs 102.50M
- YoY Growth %: 0.00
- Avg Order Value: Rs 10.25K
- Gross Margin %: 33.07

**Visuals on this page:**
- Line chart — Total Revenue by Month from January 2023 to December 2024
- Clustered bar chart — Revenue by Quarter and Year comparing 2023 vs 2024
- Donut chart — Revenue by Category showing Accessories 23.4%, Electronics 20.8%, Components 19.6%, Mechanical 19.4%, Tools 16.8%

**Slicers:** Warehouse, Category

---

### Page 2 — Sales Analysis

![Sales Analysis](<img width="1077" height="609" alt="image" src="https://github.com/user-attachments/assets/176e3c03-a9de-46a0-9f41-977b86474f88" />)

**KPI Cards:**
- Top Category: Accessories
- Best Product: Product_139
- Avg Order Value: Rs 10.25K
- Peak Day: Sunday

**Visuals on this page:**
- Horizontal bar chart — Total Revenue by Product Name (Top 8 products)
- Bar chart — Total Revenue by Day of Week
- Matrix table — Category vs Best Month Revenue, Best Product, Critical Warehouse, Gross Profit, Gross Margin %

**Key Findings:**
- Product_139 leads with Rs 605,886 in total revenue
- Sunday is the peak sales day at Rs 15.38M total
- Accessories category dominates all 5 categories

---

### Page 3 — Inventory Analysis

![Inventory Analysis](<img width="1085" height="614" alt="image" src="https://github.com/user-attachments/assets/aa3a8c31-51e6-4cea-b0f3-90fa0bcf21b9" />)

**KPI Cards:**
- Total SKUs: 500
- Reorder Alerts: 78
- Critical Warehouse: Bangalore
- Avg Stock Coverage: 3.95x

**Visuals on this page:**
- Bar chart — Reorder Alerts by Warehouse (Bangalore 18, Mumbai 17, Hyderabad 16, Chennai 14, Delhi 13)
- Clustered bar — Sum of Stock Level vs Sum of Reorder Level by Warehouse
- Donut chart — Sum of Stock Level by Location
- Detail table — Product ID, Sum of Stock Level, Stock Coverage Ratio per warehouse

**Key Finding:** 78 SKUs (15.6% of all SKUs) are below reorder level. Bangalore is the most critical warehouse with 18 SKUs at risk.

---

### Page 4 — Supplier Analysis

![Supplier Analysis](<img width="1087" height="610" alt="image" src="https://github.com/user-attachments/assets/f78d8b94-d430-4cc2-b000-38b1cef766fc" />)

**KPI Cards:**
- Total Suppliers: 50
- Avg Lead Time: 7.82 days
- Fastest Origin: USA
- Best Product: Product_139

**Visuals on this page:**
- Bar chart — Avg Lead Time by Location (Germany, Japan, India, China, USA)
- Horizontal bar — Total Suppliers by Location (China 14, India 14, Japan 9, USA 8, Germany 5)
- Donut chart — Count of Supplier Name by Warehouse Name
- Scorecard table — Location, Total Suppliers, Gross Margin %, Sum of Lead Time

**Key Finding:** USA suppliers have the fastest average lead time. 36.1% of orders exceed the 10-day fulfilment benchmark.

---

### Forecasting Setup (Page 5)

```
Step 1: Add Line chart — X axis: dim_date[full_date] set to Month
                         Y axis: [Total Revenue]

Step 2: Click chart → Analytics pane (magnifying glass icon on right side)

Step 3: Expand Forecast → toggle ON
        Units = Months
        Length = 6
        Confidence interval = 95%
        Seasonality = Auto

Step 4: Style the lines
        Actual revenue line → blue solid
        Forecast line → orange dashed
        Confidence band → light blue fill

Step 5: Add What-If Parameter for scenario planning
        Modeling → New Parameter → name "Growth Rate"
        Min: 0   Max: 20   Increment: 1   Default: 2
```

---

## Step 5 — DAX Measures

Create each measure by selecting the `fact_sales` table in Fields pane then Home → New Measure.

### Executive Overview Measures

```dax
Total Revenue =
    SUM(fact_sales[revenue])


Gross Profit =
    SUMX(
        fact_sales,
        fact_sales[revenue] - fact_sales[quantity] * RELATED(dim_product[cost])
    )


Gross Margin % =
    DIVIDE([Gross Profit], [Total Revenue], 0) * 100


Avg Order Value =
    DIVIDE(SUM(fact_sales[revenue]), COUNTROWS(fact_sales), 0)


Revenue LY =
    CALCULATE([Total Revenue], SAMEPERIODLASTYEAR(dim_date[full_date]))


YoY Growth % =
    DIVIDE([Total Revenue] - [Revenue LY], [Revenue LY], 0) * 100


Revenue YTD =
    TOTALYTD(SUM(fact_sales[revenue]), dim_date[full_date])


Revenue MTD =
    TOTALMTD(SUM(fact_sales[revenue]), dim_date[full_date])
```

### Sales Analysis Measures

```dax
Top Category =
    CALCULATE(
        SELECTEDVALUE(dim_product[category]),
        TOPN(1,
            SUMMARIZE(fact_sales, dim_product[category],
                      "Rev", SUM(fact_sales[revenue])),
            [Rev], DESC)
    )


Best Product =
    CALCULATE(
        SELECTEDVALUE(dim_product[product_name]),
        TOPN(1,
            SUMMARIZE(fact_sales, dim_product[product_name],
                      "Rev", SUM(fact_sales[revenue])),
            [Rev], DESC)
    )


Peak Day =
    CALCULATE(
        SELECTEDVALUE(dim_date[day_name]),
        TOPN(1,
            SUMMARIZE(fact_sales, dim_date[day_name],
                      "Rev", SUM(fact_sales[revenue])),
            [Rev], DESC)
    )


Revenue Rank =
    RANKX(ALL(dim_product[product_name]),
          CALCULATE(SUM(fact_sales[revenue])), , DESC, DENSE)


Revenue Contribution % =
    DIVIDE(
        SUM(fact_sales[revenue]),
        CALCULATE(SUM(fact_sales[revenue]), ALL(dim_product)),
        0
    ) * 100
```

### Inventory Measures

```dax
Total SKUs =
    DISTINCTCOUNT(fact_inventory[product_id])


Reorder Alert Count =
    COUNTROWS(
        FILTER(fact_inventory,
               fact_inventory[stock_level] <= fact_inventory[reorder_level])
    )


Reorder Alert % =
    DIVIDE([Reorder Alert Count], [Total SKUs], 0) * 100


Avg Stock Coverage =
    DIVIDE(
        AVERAGE(fact_inventory[stock_level]),
        AVERAGE(fact_inventory[reorder_level]),
        0
    )


Critical Warehouse =
    CALCULATE(
        SELECTEDVALUE(dim_warehouse[warehouse_name]),
        TOPN(1,
            SUMMARIZE(fact_inventory, dim_warehouse[warehouse_name],
                      "Alerts",
                      COUNTROWS(FILTER(fact_inventory,
                          fact_inventory[stock_level]
                              <= fact_inventory[reorder_level]))),
            [Alerts], DESC)
    )


Inventory Turnover =
    DIVIDE(SUM(fact_sales[quantity]),
           AVERAGE(fact_inventory[stock_level]), 0)


Days Inventory Outstanding =
    DIVIDE(
        AVERAGE(fact_inventory[stock_level]),
        DIVIDE(SUM(fact_sales[quantity]), 365, 0),
        0
    )
```

### Supplier Measures

```dax
Total Suppliers =
    COUNTROWS(dim_supplier)


Avg Lead Time =
    AVERAGE(dim_supplier[lead_time])


Avg Fulfilment =
    AVERAGE(fact_orders[fulfilment_days])


Fulfilment Gap Days =
    [Avg Fulfilment] - [Avg Lead Time]


Fastest Origin =
    CALCULATE(
        SELECTEDVALUE(dim_supplier[location]),
        TOPN(1,
            SUMMARIZE(dim_supplier, dim_supplier[location],
                      "AvgLT", AVERAGE(dim_supplier[lead_time])),
            [AvgLT], ASC)
    )


On Time Delivery % =
    DIVIDE(
        COUNTROWS(FILTER(fact_orders, fact_orders[fulfilment_days] <= 10)),
        COUNTROWS(fact_orders),
        0
    ) * 100


Late Delivery % =
    DIVIDE(
        COUNTROWS(FILTER(fact_orders, fact_orders[fulfilment_days] > 10)),
        COUNTROWS(fact_orders),
        0
    ) * 100
```

### Forecasting Measures

```dax
MA3 Base =
    AVERAGEX(
        DATESINPERIOD(
            dim_date[full_date],
            LASTDATE(dim_date[full_date]),
            -3,
            MONTH
        ),
        CALCULATE(SUM(fact_sales[revenue]))
    )


Adjusted Forecast =
    [MA3 Base] * (1 + 'Growth Rate'[Growth Rate Value] / 100)


Forecast Variance =
    [Total Revenue] - [MA3 Base]


Forecast Accuracy % =
    100 - (
        DIVIDE(ABS([Total Revenue] - [MA3 Base]), [Total Revenue], 0) * 100
    )


Projected Annual Revenue =
    [MA3 Base] * 12


Revenue Trend =
    IF([Total Revenue] > [Revenue LM],  1,
    IF([Total Revenue] < [Revenue LM], -1, 0))
```

---

## Step 6 — Conditional Formatting

CF measures return 1 (Green), 2 (Amber), or 3 (Red).

**How to apply to any KPI Card:**
1. Click the Card visual
2. Format pane → Callout value → Background color → toggle ON → click fx button
3. Format style = Rules → Field = select the CF measure below
4. Add rules: value 1 = background #EAF3DE (green), 2 = #FAEEDA (amber), 3 = #FCEBEB (red)
5. Click OK

**How to apply to Table rows:**
1. Click Table visual → Format pane → Cell elements
2. Select the column to colour → Background color → fx
3. Format style = Field value → select the row-level CF measure
4. Click OK

### All CF Measures

```dax
CF YoY Growth Status =
    IF([YoY Growth %] >= 5,  1,
    IF([YoY Growth %] >= 0,  2, 3))


CF Gross Margin Status =
    IF([Gross Margin %] >= 30, 1,
    IF([Gross Margin %] >= 20, 2, 3))


CF Reorder Status =
    IF([Reorder Alert %] <= 10, 1,
    IF([Reorder Alert %] <= 20, 2, 3))


CF Stock Coverage Status =
    IF([Avg Stock Coverage] >= 3, 1,
    IF([Avg Stock Coverage] >= 2, 2, 3))


CF Lead Time Status =
    IF([Avg Lead Time] <= 7,  1,
    IF([Avg Lead Time] <= 10, 2, 3))


CF Fulfilment Status =
    IF([Avg Fulfilment] <= 8,  1,
    IF([Avg Fulfilment] <= 12, 2, 3))


CF Fulfilment Gap Status =
    IF([Fulfilment Gap Days] <= 0, 1,
    IF([Fulfilment Gap Days] <= 2, 2, 3))


CF On Time Status =
    IF([On Time Delivery %] >= 90, 1,
    IF([On Time Delivery %] >= 75, 2, 3))


CF Forecast Accuracy Status =
    IF([Forecast Accuracy %] >= 85, 1,
    IF([Forecast Accuracy %] >= 70, 2, 3))


-- Row-level CF for inventory table (apply to stock_level column)
CF Row Reorder Status =
    IF(MAX(fact_inventory[stock_level])
           <= MAX(fact_inventory[reorder_level]), 3, 1)


-- Row-level CF for supplier table (apply to lead_time column)
CF Supplier Row Status =
    IF(MAX(dim_supplier[lead_time]) <= 7,  1,
    IF(MAX(dim_supplier[lead_time]) <= 10, 2, 3))
```

### CF Threshold Reference Table

| KPI Card | Green | Amber | Red |
|---|---|---|---|
| YoY Growth % | >= 5% | 0% to 4.9% | < 0% |
| Gross Margin % | >= 30% | 20% to 29.9% | < 20% |
| Reorder Alert % | <= 10% | 10% to 20% | > 20% |
| Stock Coverage Ratio | >= 3x | 2x to 2.9x | < 2x |
| Avg Lead Time | <= 7 days | 7 to 10 days | > 10 days |
| Avg Fulfilment | <= 8 days | 8 to 12 days | > 12 days |
| Fulfilment Gap Days | <= 0 days | 0.1 to 2 days | > 2 days |
| On-Time Delivery % | >= 90% | 75% to 89.9% | < 75% |
| Forecast Accuracy | >= 85% | 70% to 84.9% | < 70% |

**Colour hex codes:**

| Status | Background | Text |
|---|---|---|
| Green | #EAF3DE | #27500A |
| Amber | #FAEEDA | #633806 |
| Red | #FCEBEB | #791F1F |

---

## Step 7 — Business Insights

### Revenue Insights

- Accessories leads with Rs 23.99M at 23.4% revenue share — increase SKU depth and marketing spend in this category
- Top product Product_139 generated Rs 605,886 — analyse its pricing model and apply similar strategy to underperforming products
- Sunday is the peak sales day at Rs 15.38M total across both years — schedule weekend promotional campaigns
- January 2024 is the best single month at Rs 4.89M — leverage Q1 demand surge patterns for inventory planning

### Inventory Insights

- 78 SKUs (15.6%) are below reorder level — amber status on the CF dashboard
- Bangalore is the most critical warehouse with 18 SKUs at risk — prioritise immediate restock
- Avg stock coverage is 3.95x the reorder level — safe overall but warehouse-level action is needed
- Recommendation: implement automated purchase order triggers at 1.2x reorder level

### Supplier Insights

- USA has the fastest avg lead time at 7.1 days — increase allocation to US suppliers for fast-moving SKUs
- Germany has the highest per-supplier lead time at 8.6 days — renegotiate SLA or identify alternatives
- 36.1% of orders exceeded 10 days fulfilment — significant gap vs the promised 7.82-day avg lead time
- Recommendation: diversify to local Indian suppliers for the top 50 fastest-moving SKUs to reduce supply chain risk

### Forecast Insights

- MA(3) base forecast: Rs 4.52M per month for H1 2025
- Applying 2% monthly growth assumption: June 2025 forecast reaches Rs 4.99M
- Power BI ETS forecast with 95% confidence interval is applied on the forecasting page
- Recommendation: use the What-If growth rate slider from 0 to 20% for board presentation scenario planning

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- MySQL 8.0 running on 127.0.0.1:3306
- Power BI Desktop (free download from Microsoft)
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/supplysight.git
cd supplysight

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Create MySQL database (run this in MySQL Workbench)
# CREATE DATABASE IF NOT EXISTS supply_chain_analytics;

# 4. Run data cleaning pipeline
python python/01_data_cleaning.py

# 5. Upload cleaned data to MySQL
python python/02_mysql_integration.py

# 6. Open SQL queries in MySQL Workbench
# File -> Open SQL Script -> sql/analytics_queries.sql

# 7. Open Power BI dashboard
# Double-click powerbi/SupplySight.pbix
```

### requirements.txt

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
sqlalchemy>=2.0
mysql-connector-python>=8.1
```

### Environment Variables (Optional)

```bash
export DB_HOST=127.0.0.1
export DB_PORT=3306
export DB_USER=root
export DB_PASSWORD=your_password
export DB_NAME=supply_chain_analytics
```

### Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `getaddrinfo failed` | Using "localhost" on Windows | Change host to `127.0.0.1` |
| `1114 Table is full` | MySQL defaulting to MyISAM engine | Use `safe_upload()` which forces InnoDB |
| DAX card shows BLANK | TOPN measure needs text not number | Format pane → turn off Summarisation |
| `InterfaceError` with password | URL parser breaks on @ character | Wrap password with `quote_plus()` |
| `Revenue LY` returns BLANK | dim_date not connected to fact_sales | Set relationship in Model view |

---

## License

MIT License — free to use and adapt with attribution.

---

*SupplySight — Supply Chain and Inventory Analytics | 2026*

*https://github.com/Shridharpatil1958/Supply-Chain-Inventory-Optimization*
