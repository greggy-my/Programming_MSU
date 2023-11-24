import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

file_path = '../datasets/diamonds.csv'
df = pd.read_csv(file_path)

df = df.drop(columns=['index'])
print(df.head(5))

df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

print(df.head(5))

df = shuffle(df, random_state=42)

X = df.drop(columns=['price'])
y = df['price']

param_grid = [
    {'criterion': ['squared_error'], 'max_depth': [12]},
    {'criterion': ['friedman_mse'], 'max_depth': [16]},
    {'criterion': ['poisson'], 'max_depth': [22]},
    {'criterion': ['squared_error'], 'max_depth': [45]},
    {'criterion': ['friedman_mse'], 'max_depth': [95]},
    {'criterion': ['poisson'], 'max_depth': [33]}
]

regressor = DecisionTreeRegressor()

grid_search = GridSearchCV(regressor, param_grid, cv=10, scoring='r2', n_jobs=-1)
grid_search.fit(X, y)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучшее среднее качество (R^2):", grid_search.best_score_)
