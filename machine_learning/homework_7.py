import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(42)


class sample(object):
    def __init__(self, x, n_subspace):
        self.idx_subspace = self.random_subspace(x, n_subspace)

    def __call__(self, x, y):
        idx_obj = self.bootstrap_sample(x)
        x_sampled, y_sampled, idx_subspace = self.get_subsample(x, y, self.idx_subspace, idx_obj)
        return x_sampled, y_sampled, idx_subspace

    @staticmethod
    def bootstrap_sample(x, random_state=42):
        """
        Возвращает массив индексов выбранных при помощи бэггинга индексов.
        """
        idx_bootstrap = np.random.choice(len(x), len(x), replace=True)
        idx_bootstrap = np.unique(idx_bootstrap)
        return idx_bootstrap

    @staticmethod
    def random_subspace(x, n_subspace, random_state=42):
        """
        Возвращает массив индексов выбранных при помощи метода случайных подпространств признаков.
        """
        idx_subspace = np.random.choice(x.shape[1], n_subspace, replace=False)
        return idx_subspace

    @staticmethod
    def get_subsample(x, y, idx_subspace, idx_obj):
        """
        Возвращает подвыборку X_sampled, y_sampled по значениям индексов признаков и объектов.
        """
        x_sampled = x[idx_obj][:, idx_subspace]
        y_sampled = y[idx_obj]
        return x_sampled, y_sampled, idx_subspace


N_ESTIMATORS = 100
MAX_DEPTH = 5
SUBSPACE_DIM = 3


class random_forest(object):
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.subspaces_idx = []
        self.random_state = random_state
        self._estimators = []

    def fit(self, x, y):
        for i in range(self.n_estimators):
            # Создаем объект класса Sample для формирования подвыборок
            sampler = sample(x, self.subspaces_dim)
            # Получаем подвыборку
            x_sampled, y_sampled, subspaces_idx = sampler(x, y)

            # Создаем и обучаем дерево решений
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(x_sampled, y_sampled)

            # Добавляем дерево в список
            self._estimators.append(tree)
            self.subspaces_idx.append(subspaces_idx)

    def predict(self, x):
        # Получаем предсказания от каждого дерева в ансамбле
        predictions = [tree.predict(x[:, self.subspaces_idx[index]]) for index, tree in enumerate(self._estimators)]
        # Вычисляем среднее предсказание
        avg_prediction = np.mean(predictions, axis=0)
        # Округляем до ближайшего целого
        return np.round(avg_prediction).astype(int)


# Загрузим датасет
iris = load_iris()
x, y = iris.data, iris.target

# Разделим данные на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Создаем и обучаем случайный лес
rf = random_forest(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, subspaces_dim=SUBSPACE_DIM, random_state=42)
rf.fit(x_train, y_train)

# Получаем предсказания на тестовой выборке
y_pred = rf.predict(x_test)

# Оцениваем качество модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")