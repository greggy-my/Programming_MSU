import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import warnings

warnings.filterwarnings("ignore")

# Загрузка данных
dataset_filenames = os.listdir('../datasets/Task_10')


# Функция для визуализации результатов кластеризации
def visualize_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# Функция для проверки успешности кластеризации
def evaluate_clustering(labels, true_labels):
    correct_clusters = sum(labels == true_labels)
    accuracy = correct_clusters / len(true_labels)
    return accuracy


# Применение K-Means к каждому датасету
successful_datasets = ''

for filename in dataset_filenames:
    # Загрузка данных
    data = pd.read_csv(f'../datasets/Task_10/{filename}')
    X = data[['x', 'y']].values
    true_labels = data['class'].values

    # Применение K-Means
    kmeans = KMeans(n_clusters=len(set(true_labels)), random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    print(labels)
    print(true_labels)

    # Визуализация результатов
    visualize_clusters(X, labels, f'K-Means Clustering - {filename}')

    # Оценка успешности кластеризации
    accuracy = evaluate_clustering(labels, true_labels)
    print(f'Accuracy for {filename}: {accuracy * 100:.2f}%')

    # Проверка условия успешности (больше 90% объектов в правильных кластерах)
    if accuracy >= 0.9:
        successful_datasets += filename.split('.')[0][-1]

# Вывод успешных датасетов
print(f'Successful datasets for K-Means: {successful_datasets}')
