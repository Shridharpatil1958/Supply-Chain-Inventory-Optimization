import pandas as pd

def abc_classification(df):
    """
    Required Columns:
    Product_ID, Annual_Demand, Unit_Cost
    """

    df["Annual_Value"] = df["Annual_Demand"] * df["Unit_Cost"]

    df = df.sort_values(
        by="Annual_Value",
        ascending=False
    )

    df["Cumulative_Percentage"] = (
        df["Annual_Value"].cumsum()
        / df["Annual_Value"].sum()
    ) * 100

    def classify(x):
        if x <= 80:
            return "A"
        elif x <= 95:
            return "B"
        else:
            return "C"

    df["ABC_Class"] = df["Cumulative_Percentage"].apply(classify)

    return df
