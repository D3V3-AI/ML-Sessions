# ================================================================================
# ML Sessions #2 - House Pricing
# ================================================================================
import json

import arff
import pandas as pd

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
with open("data/house_dataset.arff", "r") as f:
    dataset = arff.load(f)

df = pd.DataFrame(
    dataset["data"],
    columns=[attr[0] for attr in dataset["attributes"]]
)



# ─────────────────────────────────────────
# DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────
df = df.dropna()

# ----- add new features ----- # 

df['rooms_per_household'] = (
    df['total_rooms'] / df['households']
)

df['population_per_household'] = (
    df['population'] / df['households']
)

df['rooms_per_person'] = (
    df['total_rooms'] / df['population']
)

# ----- log transform skewed features ----- #
for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
    df[f'log_{col}'] = df[col].apply(lambda x: np.log1p(x))

df = df.drop(['total_rooms', 'total_bedrooms', 'population', 'households'], axis=1)

# ─────────────────────────────────────────
# X & Y SPLIT
# ─────────────────────────────────────────
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────
# MODEL PIPELINE
# ─────────────────────────────────────────
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

categorial_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([    
    ("num", "passthrough", numeric_features),
    ("cat", categorial_transformer, categorical_features)
])

ml_model = XGBRegressor(
    n_estimators=100,               # number of trees
    learning_rate=0.1,              # 
    max_depth=6,                    # maximum depth of each tree       
    subsample=0.8,                  # subsample ratio of the training instances
    objective="reg:squarederror",   # objective function for regression
    random_state=42,                # 
    n_jobs=-1                       # 
)


model = Pipeline([
    ('preprocessor', preprocessor),
    ("ml_model", ml_model)
])

# ─────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────
def train():
    model.fit(X_train, y_train)

# ─────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────
def evaluate():
    y_pred = model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Root mean squared error: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")