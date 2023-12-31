import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump

df = pd.read_csv("./bank-full.csv", sep=";")

job_types = df["job"].unique()
job = df["job"].map({f"{v}": i for i, v in enumerate(job_types)})

education_types = df["education"].unique()
education = df["education"].map({f"{v}": i for i, v in enumerate(education_types)})
month_types = df["month"].unique()
month = df["month"].map({f"{v}": i for i, v in enumerate(month_types)})

default = df["default"].map({"yes": 1, "no": 0})
housing = df["housing"].map({"yes": 1, "no": 0})
loan = df["loan"].map({"yes": 1, "no": 0})
marital = df["marital"].map({"married": 1, "single": 0, "divorced": 2})
contact = df["contact"].map({"unknown": 0, "cellular": 1, "telephone": 2})
poutcome = df["poutcome"].map({"unknown": 0, "failure": 1, "other": 2, "success": 3})

y = df["y"].map({"yes": 1, "no": 0})
df.drop(
    [
        "education",
        "default",
        "month",
        "housing",
        "loan",
        "contact",
        "poutcome",
        "job",
        "marital",
        "y",
    ],
    axis=1,
    inplace=True,
)
new_data = pd.concat(
    [df, job, marital, month, education, default, housing, loan, contact, poutcome, y],
    axis=1,
)

y = new_data["y"]

new_data.drop(["y"], axis=1, inplace=True)
X = new_data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# param_grid = {"max_depth": [5, 10, 15, 20, 25]}

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [10, 15, 20, 25],
#     'min_samples_split': [2, 5, 10]
# }

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")


model = RandomForestClassifier(
    max_depth=20, n_estimators=50, min_samples_split=5, criterion='gini')
model.fit(X_train, y_train)

model_file_name = "bank.joblib"
dump(model, model_file_name)

y_predict = model.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, y_predict) * 100:.2f}%")
