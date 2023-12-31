import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./insurance.csv")

sex = pd.get_dummies(df["sex"], drop_first=True, dtype=int)
region = pd.get_dummies(df["region"], drop_first=True, dtype=int)
smoker = df["smoker"].map({"yes": 1, "no": 0})


df.drop(["sex", "smoker", "region"], axis=1, inplace=True)

new_data = pd.concat([df, sex, smoker, region], axis=1)

y = new_data["expenses"]

new_data.drop(["expenses"], axis=1, inplace=True)

X = new_data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {"max_depth": [5, 10, 15, 20, 25]}

rf = RandomForestRegressor()

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_max_depth = grid_search.best_params_['max_depth']
print(f"Best max_depth: {best_max_depth}")

class_weights = {0: 1, 1: 5}


model = RandomForestRegressor(max_depth=5, n_estimators=50)
model.fit(X_train, y_train)

model_file_name = "insurance.joblib"
dump(model, model_file_name)

test_r2_score = model.score(X_test, y_test)
print(f"Model Accuracy is
      
      : {test_r2_score * 100:.2f}%")

y_predict = model.predict(X_test)
print(f"Mean square error: {mean_squared_error(y_test, y_predict)}")


final_data = pd.DataFrame(
    {
        "age": X_test["age"].values,
        "smoker": X_test["smoker"].values,
        "charges": [*y_predict],
    }
)

# sns.lineplot(data=final_data, hue='smoker',x='age', y='charges')

custom_smoker  = final_data.loc[final_data["smoker"] == 1]
custom_none_somoker = final_data.loc[final_data["smoker"] == 0]

plt.figure(figsize=(8, 6))
plt.scatter(custom_smoker["age"], custom_smoker["charges"], color="red")
plt.scatter(custom_none_somoker["age"], custom_none_somoker["charges"], color="blue")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Smoker Vs Non smoker")
plt.legend(["Smoker", "None smoker"])
plt.grid(True)
plt.tight_layout()
plt.show()
