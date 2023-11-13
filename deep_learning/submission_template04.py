import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from IPython.display import clear_output

# загружаем датасет из torchvision
train_data = datasets.CIFAR10(root="../cifar10_data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="../cifar10_data", train=False, download=True, transform=transforms.ToTensor())

# делим тренировочную часть на train и val

# в тренировочную выборку отнесем 80% всех картинок
train_size = int(len(train_data) * 0.8)
# в валидационную — остальные 20%
val_size = len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

# заводим даталоадеры, которые будут генерировать батчи
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# функция отрисовки картинок
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


# получаем батч картинок
for batch in train_loader:
    images, labels = batch
    break


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn_1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn_2 = nn.BatchNorm2d(5)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        # Рассчитываем количество входных нейронов для fc1
        self.fc1_in_features = 11 * 11 * 5  # После применения сверточных слоев и maxpool2 изображение будет размером 5x5
        self.fc1 = nn.Linear(self.fc1_in_features, 250)
        self.bn_3 = nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(250, 150)
        self.bn_4 = nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(150, 50)
        self.bn_5 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        # Реализация forward pass сети
        x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        x = self.bn_1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = self.bn_2(x)
        x = F.relu(self.conv3(x))
        # x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bn_3(x)
        x = F.relu(self.fc2(x))
        x = self.bn_4(x)
        x = F.relu(self.fc3(x))
        x = self.bn_5(x)
        x = self.fc4(x)
        return x


def evaluate(model, dataloader, loss_fn):
    losses = []

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):
        # так получаем текущий батч
        X_batch, y_batch = batch
        num_elements += len(y_batch)

        with torch.no_grad():
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1)

            num_correct += torch.sum(y_pred.cpu() == y_batch)

    accuracy = num_correct / num_elements

    return accuracy.numpy(), np.mean(losses)


def train(model, loss_fn, optimizer, n_epoch=10):
    # цикл обучения сети
    for epoch in tqdm(range(n_epoch)):

        print("Epoch:", epoch + 1)

        model.train(True)

        running_losses = []
        running_accuracies = []
        for i, batch in enumerate(train_loader):
            # так получаем текущий батч
            X_batch, y_batch = batch

            # forward pass (получение ответов на батч картинок)
            logits = model(X_batch.to(device))

            # вычисление лосса от выданных сетью ответов и правильных ответов на батч
            loss = loss_fn(logits, y_batch.to(device))
            running_losses.append(loss.item())

            loss.backward()  # backpropagation (вычисление градиентов)
            optimizer.step()  # обновление весов сети
            optimizer.zero_grad()  # обнуляем веса

            # вычислим accuracy на текущем train батче
            model_answers = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_batch == model_answers.cpu()) / len(y_batch)
            running_accuracies.append(train_accuracy)

            # Логирование результатов
            if (i + 1) % 100 == 0:
                print("Средние train лосс и accuracy на последних 50 итерациях:",
                      np.mean(running_losses), np.mean(running_accuracies), end='\n')

        # после каждой эпохи получаем метрику качества на валидационной выборке
        model.train(False)

        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn=loss_fn)
        print("Эпоха {}/{}: val лосс и accuracy:".format(epoch + 1, n_epoch, ),
              val_loss, val_accuracy, end='\n')

    return model

# снова объявим модель
model = ConvNet()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# выбираем функцию потерь
loss_fn = torch.nn.CrossEntropyLoss()

# выбираем алгоритм оптимизации и learning_rate.
# вы можете экспериментировать с разными значениями learning_rate
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# запустим обучение модели
# параметр n_epoch можно варьировать
model = train(model, loss_fn, optimizer, n_epoch=10)

test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print('Accuracy на тесте', test_accuracy)

if test_accuracy <= 0.5:
    print("Качество на тесте ниже 0.5, 0 баллов")
elif test_accuracy < 0.6:
    print("Качество на тесте между 0.5 и 0.6, 0.5 баллов")
elif test_accuracy >= 0.6:
    print("Качество на тесте выше 0.6, 1 балл")

    model.eval()
    x = torch.randn((1, 3, 32, 32))
    torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model_04.pth")