import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm as tqdm
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification


# 1
# random_seed = 4238
# np.random.seed(random_seed)
#
# X, y = load_breast_cancer(return_X_y=True)
# X_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, shuffle=True, random_state=42
# )
#
# clf = KNeighborsClassifier(n_neighbors=8, p=1)
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(x_test)
# acc = accuracy_score(y_true=y_test, y_pred=predictions)
#
# print(acc)


# 2
# n_splits = 3
#
# X, y = load_breast_cancer(return_X_y=True)
#
# """
#   Здесь Вам предлагается написать тело цикла для подбора оптимального K
#   Результаты оценки алгоритма при каждом отдельно взятом K рекомендуем записывать в список cv_scores
# """
# cv_scores = []
# for k in tqdm(range(1, 51)):
#     clf = KNeighborsClassifier(n_neighbors=k)
#     score = cross_val_score(clf, X, y, cv=3)
#     cv_scores.append(np.mean(score))
#     print(np.mean(score))
#
# print(np.argmax(cv_scores))


# 3

class KNN_classifier:
    def __init__(self, n_neighbors: int, **kwargs):
        self.K = n_neighbors

    def fit(self, x: np.array, y: np.array):
        # Сохраняем обучающие данные и метки
        self.X_train = x
        self.y_train = y

    def predict(self, x: np.array):
        predictions = []

        for test_point in x:
            # Рассчитываем расстояния от объекта test_point до всех точек обучающей выборки
            distances = [np.linalg.norm(test_point - train_point) for train_point in self.X_train]

            # Получаем индексы K ближайших соседей
            k_nearest_indices = np.argsort(distances)[:self.K]

            # Извлекаем метки этих соседей
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Выбираем наиболее часто встречающийся класс среди соседей
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions


# Создадим синтетический набор данных для классификации
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# Разделим данные на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Используем реализацию KNN из scikit-learn для сравнения
sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)
sklearn_predictions = sklearn_knn.predict(X_test)

# Используем нашу реализацию KNN
our_knn = KNN_classifier(n_neighbors=3)
our_knn.fit(X_train, y_train)
our_predictions = our_knn.predict(X_test)

# Сравним точность (accuracy) двух реализаций
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
our_accuracy = accuracy_score(y_test, our_predictions)

print(f"Точность KNN из scikit-learn: {sklearn_accuracy:.5f}")
print(f"Точность нашей реализации KNN: {our_accuracy:.5f}")
