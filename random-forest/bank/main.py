import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./bank-full.csv", sep=";")

colums = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "y",
]
for c in colums:
    c_unique = sorted(df[c].unique())
    df[c] = df[c].map({f"{k}": v for v, k in enumerate(c_unique)})

y = df["y"]
X = df.drop("y", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rfc = RandomForestClassifier(
    n_estimators=50, max_depth=20, criterion="gini", min_samples_split=20
)

rfc.fit(X_train, y_train)

y_predict = rfc.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, y_predict) * 100:.2f}%")

final_data = X_test.copy()
final_data["y"] = y_predict
final_data["y"] = final_data["y"].map({0: "No", 1: "Yes"})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=final_data, x="age", y="balance", hue="y", palette="Set1")
plt.xlabel("Age")
plt.ylabel("Balance")
plt.title("Age and Balance with Prediction Labels")
plt.legend(title="Deposit")
plt.grid()
plt.show()
