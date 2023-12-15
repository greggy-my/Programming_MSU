import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from torchvision import transforms
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

labels = {
    0: 'background',
    1: 'car',
    2: 'wheel',
    3: 'light',
    4: 'windows'
}

images = []
segmentation_masks = []

for img_name in os.listdir('../datasets/car-segmentation/images'):
    images.append('../datasets/car-segmentation/images/' + img_name)

for img_name in os.listdir('../datasets/car-segmentation/masks'):
    segmentation_masks.append('../datasets/car-segmentation/masks/' + img_name)

images = sorted(images)
segmentation_masks = sorted(segmentation_masks)


class CarsDataset(Dataset):
    def __init__(self, images, segmentation_masks):
        self.images = images
        self.segmentation_masks = segmentation_masks

        self.images_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # размер каждой картинки будет приведен к 256*256
            transforms.ToTensor(),  # картинка из питоновского массива переводится в формат torch.Tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # значения пикселей картинки нормализуются
        ])

        self.masks_transforms = transforms.Compose([
            # используем InterpolationMode.NEAREST, чтобы при изменении размера
            # маски сегментации не менялись номера классов
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            # размер каждой картинки будет приведен к 256*256
            transforms.ToTensor(),  # картинка из питоновского массива переводится в формат torch.Tensor
        ])

    def __getitem__(self, index):
        '''
        этот метод должен по заданному номеру пары картинка-сегментация (index)
        возвращать эту пару. Этот метод обязательно нужно реализовывать во всех
        кастомных классах Dataset. Перед тем, как выдать на выходе
        пару картинка-сегментация, можно применить к картинкам любые преобразования:
        например, знакомую нам аугментацию.
        '''

        # загружаем нужные картинку и ее карту сегментации
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.segmentation_masks[index])

        # # применяем предобработку к RGB картинке
        img = self.images_transforms(img)
        # # применяем предобработку к маске сегментации
        mask = self.masks_transforms(mask) * 255

        # # делим маску сегментации на 13 бинарных масок сегментации
        # # отдельно для каждого класса
        masks = []

        # вытаскиваем пиксели, принадлежащие классам unlabeled, building, road и car.
        for i in range(5):
            # генерируем бинарную маску сегментации для текущего класса i
            cls_mask = torch.where(mask == i, 1, 0)
            masks.append(cls_mask[0, :, :].numpy())

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # схлопываем бинарные карты сегментации в одну
        masks = torch.argmax(masks, axis=0)

        # возвращаем пару: картинка — ее маска сегментации на 5 классов
        return (img, masks)

    def __len__(self):
        '''
        этот метод должен возвращать количество пар картинка-сегментация в датасете
        '''
        return len(self.images)


dataset = CarsDataset(images, segmentation_masks)

# dataset[0] — это вызов метода __getitem__(0)
img, mask = dataset[3]

print(img.shape, mask.shape)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

g_cpu = torch.Generator().manual_seed(8888)
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g_cpu)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)


def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> max_pooling.
    # Параметры слоя conv заданы параметрами функции encoder_block.
    # MaxPooling должен быть с ядром размера 2.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )
    return block


def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> upsample.
    # Параметры слоя conv заданы параметрами функции encoder_block.
    # Upsample должен быть со scale_factor=2. Тип upsampling (mode) можно выбрать любым.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    )

    return block


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        параметры:
            - in_channels: количество каналов входного изображения
            - out_channels: количество каналов выхода нейросети
        '''
        super().__init__()

        self.enc1_block = encoder_block(in_channels, 32, 7, 3)
        self.enc2_block = encoder_block(32, 64, 3, 1)
        self.enc3_block = encoder_block(64, 128, 3, 1)

        # поймите, какие параметры должны быть у dec1_block, dec2_block и dec3_block
        # dec1_block должен быть симметричен блоку enc3_block
        # dec2_block должен быть симметричен блоку enc2_block
        # но обратите внимание на skip connection между выходом enc2_block и входом dec2_block
        # (см что подается на вход dec2_block в функции forward)
        # какое количество карт активации будет тогда принимать на вход dec2_block?
        # также обратите внимание на skip connection между выходом enc1_block и входом dec3_block
        # (см что подается на вход dec3_block в функции forward)
        # какое количество карт активации будет тогда принимать на вход dec3_block?
        self.dec1_block = decoder_block(128, 64, 3, 1)
        self.dec2_block = decoder_block(128, 32, 3, 1)
        self.dec3_block = decoder_block(64, out_channels, 3, 1)

    def forward(self, x):
        # downsampling part
        enc1 = self.enc1_block(x)
        enc2 = self.enc2_block(enc1)
        enc3 = self.enc3_block(enc2)

        dec1 = self.dec1_block(enc3)
        # из-за skip connection dec2 должен принимать на вход сконкатенированные карты активации
        # из блока dec1 и из блока enc2.
        # конкатенация делается с помощью torch.cat
        dec2 = self.dec2_block(torch.cat([dec1, enc2], 1))
        # из-за skip connection dec3 должен принимать на вход сконкатенированные карты активации
        # из блока dec2 и из блока enc1.
        # конкатенация делается с помощью torch.cat
        dec3 = self.dec3_block(torch.cat([dec2, enc1], 1))

        return dec3


def train(model, opt, loss_fn, epochs, train_loader, val_loader):
    for epoch in tqdm(range(epochs)):

        # 1. Обучаем сеть на картинках из train_loader
        model.train()  # train mode

        avg_train_loss = 0
        for i, (X_batch, Y_batch) in enumerate(train_loader):
            # переносим батч на GPU
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # получаем ответы сети на батч
            Y_pred = model(X_batch)

            # считаем лосс, делаем шаг оптимизации сети
            loss = loss_fn(Y_pred, Y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()

            avg_train_loss += loss / len(train_loader)

        # выводим средний лосс на тренировочной выборке за эпоху
        print('\navg train loss: %f' % avg_train_loss)

        # 2. Тестируем сеть на картинках из val_loader
        model.eval()

        avg_val_loss = 0
        for i, (X_batch, Y_batch) in enumerate(val_loader):
            # переносим батч на GPU
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # получаем ответы сети на батч
            Y_pred = model(X_batch)
            # считаем лосс на батче
            loss = loss_fn(Y_pred, Y_batch)

            avg_val_loss += loss / len(val_loader)

        # выводим средний лосс на валидационных данных
        print('avg val loss: %f' % avg_val_loss)

        # # 3. Визуализируем ответы сети на шести картинках из валидационных данных
        #
        # # получаем один батч из data_val
        # X_val, Y_val = next(iter(val_loader))
        # # получаем ответ сети на картинки из батча
        # Y_pred = model(X_val.to(device))
        # Y_hat = Y_pred.detach().cpu().numpy()
        # Y_hat = np.argmax(Y_hat, axis=1)
        #
        # # удаляем предыдущую визуализацию
        # clear_output(wait=True)

        # # визуализируем исходные картинки, верный ответ и ответ нашей модели
        # # визуализация ответов сети
        # clear_output(wait=True)
        # _, axes = plt.subplots(7, 6, figsize=((5 + 2) * 4, 7 * 4))
        # for k in range(6):
        #
        #     # отрисовываем 6 картинок, поданных на вход сети
        #     # картинки нормализованы, поэтому могут выглядеть непривычно
        #     axes[0][k].imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray', aspect='auto')
        #     axes[0][k].title.set_text('Input')
        #
        #     # отрисовываем правильную маску сегментации для шести картинок выше
        #     axes[1][k].imshow(Y_val[k].numpy(), cmap='gray', aspect='auto')
        #     axes[1][k].title.set_text('Real Map')
        #
        #     # отрисовываем ответы сети для каждого из пяти классов сегментации в отдельности
        #     for j in range(5):
        #         axes[j + 2][k].imshow(np.where(Y_hat[k] == j, 1, 0), cmap='gray', aspect='auto')
        #         axes[j + 2][k].title.set_text('Output for {}'.format(labels[j]))
        # plt.suptitle('%d / %d - loss: %f' % (epoch + 1, epochs, avg_val_loss))
        # plt.show()


# подумайте: какие параметры in_channels, out_channels должны быть у нашей модели?
model = UNet(3, 5).to(device)
# так как сегментация многоклассовая, обучаем с помощью кросс-энтропии
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# запускаем обучение на 10 эпох
train(model, opt, loss, 25, train_loader, val_loader)

# наша задача — многоклассовая сегментация, поэтому параметрtask='multiclass'
# также нужно задать количество классов — в нашем случае их 5
jaccard = JaccardIndex(task='multiclass', num_classes=5).to(device)


def evaluate(model, dataloader, loss_fn, metric_fn):
    losses = []
    metrics = []

    for i, batch in enumerate(dataloader):
        # получаем текущий батч
        X_batch, y_batch = batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        model.eval()
        with torch.no_grad():
            # получаем выход модели
            logits = model(X_batch)

            # считаем лосс по батчу
            loss = loss_fn(logits, y_batch)
            losses.append(loss.item())

            # считаем метрику по батчу
            metric = metric_fn(logits, y_batch)
            metrics.append(metric.item())

    # возвращает средние значения лосса и метрики
    return np.mean(losses), np.mean(metrics)


print(evaluate(model, val_loader, loss, jaccard))


def get_predictions(model, dataloader):

    predictions = []

    for i, batch in enumerate(dataloader):

        X_batch, y_batch = batch
        X_batch = X_batch.to(device)

        with torch.no_grad():
            logits = model(X_batch)
            Y_hat = logits.detach().cpu().numpy()
            Y_hat = np.argmax(Y_hat, axis=1)
            predictions.extend(Y_hat)

    return predictions


predicted_labels = get_predictions(model, val_loader)

np.save('submission_hw09.npy', predicted_labels, allow_pickle=True)
print('Ответ сохранен в файл `submission_hw09.npy`')