"""
Housing Price per Sqft Prediction (Delhi NCR)

- Models: Linear Regression, Decision Tree, Random Forest
- Feature engineering: City extraction from Address
- Hyperparameter tuning: RandomizedSearchCV
- Evaluation: RMSE on hold-out test set

Author: Trijal Mohan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

print("Loading dataset")
data = pd.read_csv("data/Delhi_v2.csv")
print("Dataset loaded")
print("Shape:", data.shape)
print("Columns:")
print(data.columns)
print("Sample rows:")
print(data.head())
print("Creating stratified train-test split")

data["price_cat"] = np.ceil(data["Price_sqft"] / 2000)
data["price_cat"] = data["price_cat"].clip(upper=5)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(data, data["price_cat"]):
    strat_train_set = data.loc[train_idx].drop("price_cat", axis=1)
    strat_test_set = data.loc[test_idx].drop("price_cat", axis=1)

print("Stratified split done")
print("Train size:", len(strat_train_set))
print("Test size:", len(strat_test_set))
print("Separating features and labels...")

X_train = strat_train_set.drop("Price_sqft", axis=1)
y_train = strat_train_set["Price_sqft"].copy()

X_test = strat_test_set.drop("Price_sqft", axis=1)
y_test = strat_test_set["Price_sqft"].copy()

print("Features & labels separated")

class CityExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def extract_city(address):
            parts = str(address).lower().split(",")
            for p in reversed(parts):
                if "gurgaon" in p:
                    return "Gurgaon"
                if "greater noida" in p:
                    return "Greater Noida"
                if "noida" in p:
                    return "Noida"
                if "ghaziabad" in p:
                    return "Ghaziabad"
                if "delhi" in p:
                    return "Delhi"
            return "Unknown"

        X["City"] = X["Address"].apply(extract_city)
        return X

num_attribs = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "price" in num_attribs:
    num_attribs.remove("price")
if "Unnamed: 0" in num_attribs:
    num_attribs.remove("Unnamed: 0")

cat_attribs = ["City"]

print("Numerical features:", num_attribs)
print("Categorical features:", cat_attribs)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# Separate pipelines for comparison
# lin_pipeline = Pipeline([
#     ("city_extractor", CityExtractor()),
#     ("preprocessing", preprocessing),
#     ("model", LinearRegression())
# ])
# tree_pipeline = Pipeline([
#     ("city_extractor", CityExtractor()),
#     ("preprocessing", preprocessing),
#     ("model", DecisionTreeRegressor(
#         max_depth=10,
#         min_samples_leaf=20,
#         random_state=42
#     ))

# ])
#pipeline for Random forest 
rf_pipeline = Pipeline([
    ("city_extractor", CityExtractor()),
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ))
]) #this here is redunt dont need to do it randomsearch does it intenrally but u can play and fine tune it as you see fit
# params_grid = {
#     "model__n_estimators" : [100,200],
#     "model__max_depth" : [10,15],
#     "model__min_samples_leaf" : [5,10]
params_distribution = {
    "model__n_estimators":randint(100,500),
    "model__max_depth":randint(8,25),
    "model__min_samples_leaf":randint(1,20),
    "model__max_features":["sqrt","log2",None]
}
random_search = RandomizedSearchCV(
    rf_pipeline,
    params_distribution,
    n_iter=20,
    cv=5,
    scoring = "neg_mean_squared_error",
    n_jobs=-1,
)
random_search.fit(X_train,y_train)
best_model = random_search.best_estimator_
# }
# grid_search = GridSearchCV(
#     rf_pipeline,
#     params_grid,
#     cv=5,
#     scoring="neg_mean_squared_error",
#     n_jobs=-1,
# )
# grid_search.fit(X_train,y_train)
# grid_search.best_params_
# best_model = grid_search.best_estimator_
# lin_pipeline.fit(X_train, y_train)
# tree_pipeline.fit(X_train, y_train)
print("Models trained")

# RMSE on full test set
final_predictions = best_model.predict(X_test)
# lin_test_pred = lin_pipeline.predict(X_test)
# tree_test_pred = tree_pipeline.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
# lin_test_rmse = np.sqrt(mean_squared_error(y_test, lin_test_pred))
# tree_test_rmse = np.sqrt(mean_squared_error(y_test, tree_test_pred))
print("Best params:", random_search.best_params_)
print("Final Test RMSE:", final_rmse)
# print(f"Linear Regression Test RMSE: {lin_test_rmse}")
# print(f"Decision Tree Test RMSE: {tree_test_rmse}")