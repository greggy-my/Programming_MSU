import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from IPython.display import clear_output

resnet_transforms = transforms.Compose([
    transforms.Resize(256),  # размер каждой картинки будет приведен к 256*256
    transforms.CenterCrop(224),  # у картинки будет вырезан центральный кусок размера 224*224
    transforms.ToTensor(),  # картинка из питоновского массива переводится в формат torch.Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # значения пикселей картинки нормализуются
])

train_data = datasets.ImageFolder('simpsons_data/train', transform=resnet_transforms)
test_data = datasets.ImageFolder('simpsons_data/test', transform=resnet_transforms)

class_to_idx = train_data.class_to_idx

# в тренировочную выборку отнесем 80% всех картинок
train_size = int(len(train_data) * 0.8)
# в валидационную — остальные 20%
val_size = len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

for batch in test_loader:
    # батч картинок и батч ответов к картинкам
    images, labels = batch
    break


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


# show_images(images, labels)


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
        self.fc1_in_features = 12 * 12 * 5  # После применения сверточных слоев и maxpool2 изображение будет размером 5x5
        self.fc1 = nn.Linear(self.fc1_in_features, 720)
        self.bn_3 = nn.BatchNorm1d(720)
        self.fc2 = nn.Linear(720, 720)
        self.bn_4 = nn.BatchNorm1d(720)
        self.fc3 = nn.Linear(720, 360)
        self.bn_5 = nn.BatchNorm1d(360)
        self.fc4 = nn.Linear(360, 42)

    def forward(self, x):
        # Реализация forward pass сети
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn_1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn_2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
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


def get_predictions(model, dataloader):
    predicted_labels = []
    model.eval()
    predicted_labels = []

    for i, batch in enumerate(dataloader):
        # так получаем текущий батч
        X_batch, _ = batch

        with torch.no_grad():
            logits = model(X_batch.to(device))
            y_pred = torch.argmax(logits, dim=1)
            predicted_labels.append(y_pred)

    predicted_labels = torch.cat(predicted_labels)
    return predicted_labels


# model — переменная, в которой находится ваша модель.
predicted_labels = get_predictions(model, test_loader)
idx_to_class = {y: x for x, y in class_to_idx.items()}
predicted_labels = [idx_to_class[x] for x in predicted_labels.data.cpu().numpy()]

np.save('submission_hw05.npy', predicted_labels, allow_pickle=True)
print('Ответ сохранен в файл `submission_hw05.npy`')
