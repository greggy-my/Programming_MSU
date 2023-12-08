from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import numpy as np

np.random.seed(42)

data = fetch_openml(data_id=42608, parser='auto')
X, y = data['data'].drop(columns='Outcome').values, data['data']['Outcome'].astype(int).values

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True)

# Предобработка данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(x_test)

# Обучение и оценка алгоритмов
tree.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

# Предсказания вероятностей класса 1 (необходимо для ROC AUC и PR AUC)
y_prob_tree = tree.predict_proba(x_test_scaled)[:, 1]
y_prob_lr = lr.predict_proba(x_test_scaled)[:, 1]
y_prob_knn = knn.predict_proba(x_test_scaled)[:, 1]
y_prob_svm = svm.predict_proba(x_test_scaled)[:, 1]

# Подсчет метрик через 2 этапа
pr= []
precision, recall, thresholds = precision_recall_curve(y_true=y_test, probas_pred=y_prob_tree)
tree_auc_pr = auc(recall, precision)
pr.append(tree_auc_pr)

precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_prob_lr)
lr_auc_pr = auc(recall, precision)
pr.append(lr_auc_pr)

precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_prob_knn)
knn_auc_pr = auc(recall, precision)
pr.append(knn_auc_pr)

precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_prob_svm)
svm_auc_pr = auc(recall, precision)
pr.append(svm_auc_pr)

print(f'PR AUC Scores: {pr}')
# Вычисление метрик
roc_auc_scores = [roc_auc_score(y_true=y_test, y_score=prob) for prob in [y_prob_tree, y_prob_lr, y_prob_knn, y_prob_svm]]
pr_auc_scores = [average_precision_score(y_true=y_test, y_score=prob) for prob in [y_prob_tree, y_prob_lr, y_prob_knn, y_prob_svm]]

# Находим лучшие алгоритмы
best_roc_auc_algorithm = roc_auc_scores.index(max(roc_auc_scores)) + 1
best_pr_auc_algorithm = pr_auc_scores.index(max(pr_auc_scores)) + 1

# Выводим результаты
print(f'ROC AUC Scores: {roc_auc_scores}')
print(f'PR AUC Scores: {pr_auc_scores}')
print(f'Best Algorithm (ROC AUC): {best_roc_auc_algorithm}')
print(f'Best Algorithm (PR AUC): {best_pr_auc_algorithm}')


def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))

    if true_positives + false_positives == 0:
        return 0.0
    else:
        return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    if true_positives + false_negatives == 0:
        return 0.0
    else:
        return true_positives / (true_positives + false_negatives)


def f1(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)

    if precision_val + recall_val == 0:
        return 0.0
    else:
        return 2 * (precision_val * recall_val) / (precision_val + recall_val)
