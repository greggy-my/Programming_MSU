import numpy as np
from sklearn.metrics import r2_score
import os

np.random.seed(42)


class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None

    def fit(self, x: np.array, y: np.array):
        # Добавляем столбец единиц для учета смещения (intercept)
        x = np.column_stack((x, np.ones(x.shape[0])))

        # Решаем уравнение для коэффициентов (параметров) модели
        self.coef_ = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x: np.array):
        # Добавляем столбец единиц для учета смещения (intercept)
        x = np.column_stack((x, np.ones(x.shape[0])))

        # Предсказываем значения с использованием обученных коэффициентов
        y_pred = x @ self.coef_
        return y_pred


def r2(y_true, y_pred):
    # Вычисляем дисперсию ошибки модели
    D_epsilon = np.var(y_true - y_pred, ddof=1)

    # Вычисляем дисперсию зависимой переменной y
    D_y = np.var(y_true, ddof=1)

    # Вычисляем коэффициент детерминации
    r_squared = 1 - D_epsilon / D_y

    return r_squared


files_list = os.listdir('../datasets/Task_3')
best_model_score = -np.inf
worst_model_score = np.inf
for index, file in enumerate(files_list):
    path = f'../datasets/Task_3/{file}'
    data = np.load(path)
    y = data[:, 1:]
    x = data[:, 0:1]

    LinReg = LinearRegression()
    LinReg.fit(x, y)
    prediction = LinReg.predict(x)
    model_score = r2(y, prediction)

    if model_score > best_model_score:
        best_model_score = model_score
        best_model = file

    if model_score < worst_model_score:
        worst_model_score = model_score
        worst_model = file

    print(model_score)
    print(best_model, worst_model)

# Вывод результатов (индексы начинаются с 1, поэтому добавляем 1)
print(f"Лучшая выборка: {best_model}")
print(f"Наихудшая выборка: {worst_model}")