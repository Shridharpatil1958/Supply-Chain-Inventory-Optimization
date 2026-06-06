import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data/inventory.csv")

X = df[
    [
        "Current_Stock",
        "Lead_Time",
        "Avg_Daily_Demand"
    ]
]

y = df["Stockout"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

joblib.dump(
    model,
    "models/stockout_model.pkl"
)
