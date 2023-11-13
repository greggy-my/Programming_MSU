import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from IPython.display import clear_output
from torch.nn.modules.batchnorm import BatchNorm1d
from sklearn.metrics import accuracy_score


# эта функция отрисовывает несколько картинок
def show_images(images, labels):
    f, axes = plt.subplots(1, 10, figsize=(30, 5))

    for i, axis in enumerate(axes):
        # переводим картинку из тензора в numpy
        img = images[i].numpy()
        # переводим картинку в размерность (длина, ширина, цветовые каналы)
        img = np.transpose(img, (1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(labels[i].numpy())

    plt.show()


def evaluate(model, dataloader, loss_fn):
    y_pred_list = []
    y_true_list = []
    losses = []

    # проходимся по батчам даталоадера
    for i, batch in enumerate(dataloader):
        # так получаем текущий батч
        X_batch, y_batch = batch

        # выключаем подсчет любых градиентов
        with torch.no_grad():
            # получаем ответы сети на батч
            logits = model(X_batch.to(device))

            # вычисляем значение лосс-функции на батче
            loss = loss_fn(logits, y_batch.to(device))
            loss = loss.item()

            # сохраняем лосс на текущем батче в массив
            losses.append(loss)

            # для каждого элемента батча понимаем,
            # к какому классу от 0 до 9 отнесла его сеть
            y_pred = torch.argmax(logits, dim=1)

        # сохраняем в массивы правильные ответы на текущий батч
        # и ответы сети на текущий батч
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

    # считаем accuracy между ответам сети и правильными ответами
    accuracy = accuracy_score(y_pred_list, y_true_list)

    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, n_epoch=6):
    model.train(True)

    data = {
        'acc_train': [],
        'loss_train': [],
        'acc_val': [],
        'loss_val': []
    }

    # цикл обучения сети
    for epoch in tqdm(range(n_epoch)):

        for i, batch in enumerate(train_loader):
            # так получаем текущий батч картинок и ответов к ним
            X_batch, y_batch = batch

            # forward pass (получение ответов сети на батч картинок)
            logits = model(X_batch.to(device))

            # вычисление лосса от выданных сетью ответов и правильных ответов на батч
            loss = loss_fn(logits, y_batch.to(device))

            optimizer.zero_grad()  # обнуляем значения градиентов оптимизаторв
            loss.backward()  # backpropagation (вычисление градиентов)
            optimizer.step()  # обновление весов сети

        # конец эпохи, валидируем модель
        print('On epoch end', epoch)

        acc_train_epoch, loss_train_epoch = evaluate(model, train_loader, loss_fn)
        print('Train acc:', acc_train_epoch, 'Train loss:', loss_train_epoch)

        acc_val_epoch, loss_val_epoch = evaluate(model, val_loader, loss_fn)
        print('Val acc:', acc_val_epoch, 'Val loss:', loss_val_epoch)

        data['acc_train'].append(acc_train_epoch)
        data['loss_train'].append(loss_train_epoch)
        data['acc_val'].append(acc_val_epoch)
        data['loss_val'].append(loss_val_epoch)

    return model, data


# Реализуйте модель.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc_in = nn.Linear(3072, 1024)
        self.bn_1 = BatchNorm1d(1024)
        self.fc_hidden_1 = nn.Linear(1024, 512)
        self.bn_2 = BatchNorm1d(512)
        self.fc_hidden_2 = nn.Linear(512, 256)
        self.bn_3 = BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc_in(x))
        x = self.bn_1(x)
        x = F.relu(self.fc_hidden_1(x))
        x = self.bn_2(x)
        x = F.relu(self.fc_hidden_2(x))
        x = self.bn_3(x)
        x = self.fc_out(x)
        return x


train_data = datasets.CIFAR10(root="../cifar10_data", train=True, download=False, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="../cifar10_data", train=False, download=False, transform=transforms.ToTensor())

# разделим тренировочную выборку на train и val
# в тренировочную выборку отнесем 80% всех картинок
train_size = int(len(train_data) * 0.8)
# в валидационную — остальные 20%
val_size = len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

# заведем даталоадеры
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# получаем батч картинок из train даталоадера
for batch in train_loader:
    # батч картинок и батч ответов к картинкам
    images, labels = batch
    break

# # вызываем функцию отрисовки картинок
# show_images(images, labels)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

# объявляем модель и переносим ее на GPU
model = Model().to(device)

assert model is not None, 'Переменная model пустая. Где же тогда ваша модель?'

try:
    x = images.reshape(-1, 3072).to(device)
    y = labels

    # compute outputs given inputs, both are variables
    y_predicted = model(x)
except Exception as e:
    print('С моделью что-то не так')
    raise e

assert y_predicted.shape[-1] == 10, 'В последнем слое модели неверное количество нейронов'

# объявляем модель и переносим ее на GPU
model = Model().to(device)

# функция потерь
loss_fn = torch.nn.CrossEntropyLoss()

# оптимизатор
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model, data = train(model, loss_fn, optimizer, n_epoch=20)

test_acc, test_loss = evaluate(model, test_loader, loss_fn)

print(test_acc)

assert test_acc >= 0.5, 'Accuracy на тесте >0.5! Можно сдавать задание'

if test_acc > 0.5:
    x = torch.randn((64, 32 * 32 * 3))
    torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model_03.pth")
