import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from IPython.display import clear_output
import resource
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, 100000))
print(device)

images = []


for dirpath, dirnames, filenames in os.walk('../datasets/lfw-deepfunneled'):
    for fname in filenames:
        if fname.endswith(".jpg"):
            fpath = os.path.join(dirpath, fname)
            img = Image.open(fpath)
            images.append(img)


# def plot_gallery(images, n_row=3, n_col=6, from_torch=False):
#     """Helper function to plot a gallery of portraits"""
#
#     # нужно поставить from_torch=True, если функция plot_gallery
#     # вызывается для images типа torch.Tensor
#     if from_torch:
#         images = [x.data.numpy().transpose(1, 2, 0) for x in images]
#
#     plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i])
#
#         # убираем отрисовку координат
#         plt.xticks(())
#         plt.yticks(())
#
#     plt.show()
#
#
# plot_gallery(images)


class Faces(Dataset):
    def __init__(self, faces):
        self.data = faces
        self.transform = transforms.Compose([
            transforms.CenterCrop((90, 90)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        x = self.data[index]
        return self.transform(x).float()

    def __len__(self):
        return len(self.data)


dataset = Faces(images)

# dataset[0] — это вызов метода __getitem__(0)
img = dataset[0]

print(img.shape)

# # отрисовываем несколько картинок
# plot_gallery(dataset, from_torch=True)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

g_cpu = torch.Generator().manual_seed(8888)
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g_cpu)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)


def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
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
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    )

    return block


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()


        # добавьте несколько слоев encoder block
        # это блоки-составляющие энкодер-части сети
        self.encoder = nn.Sequential(
            encoder_block(3, 32, 7, 3),
            encoder_block(32, 64, 3, 1),
            encoder_block(64, 128, 3, 1)
        )

        # добавьте несколько слоев decoder block
        # это блоки-составляющие декодер-части сети
        self.decoder = nn.Sequential(
            decoder_block(128, 64, 3, 1),
            decoder_block(64, 32, 3, 1),
            decoder_block(32, 3, 3, 1)
        )

    def forward(self, x):

        # downsampling
        latent = self.encoder(x)

        # upsampling
        reconstruction = self.decoder(latent)

        return reconstruction


# проверка, что у модели есть обучаемые слои
model = Autoencoder()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in model_parameters])
assert num_params > 10

# проверка, что модель собрана верно
random_tensor = torch.Tensor(np.random.random((32, 3, 64, 64)))
model = Autoencoder()
out = model(random_tensor)
assert out.shape == (32, 3, 64, 64), "неверный размер выхода модели"

# проверка, что у модели можно отцепить декодер и использовать его как
# отдельную модель
# если здесь возникла ошибка, убедитесь, что в вашей сети нет skip connection
random_tensor = torch.Tensor(np.random.random((32, 3, 64, 64)))
model = Autoencoder()
latent_shape = model.encoder(random_tensor).shape
latent = torch.Tensor(np.random.random(latent_shape))
out = model.decoder(latent)

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def train(model, opt, loss_fn, epochs, train_loader, val_loader):
    for epoch in tqdm(range(epochs)):

        # 1. Обучаем сеть на картинках из train_loader
        model.train()  # train mode

        avg_train_loss = 0
        for i, X_batch in enumerate(train_loader):
            # переносим батч на GPU
            X_batch = X_batch.to(device)
            # получаем ответы сети на батч
            Y_pred = model(X_batch)

            # считаем лосс, делаем шаг оптимизации сети
            loss = loss_fn(Y_pred, X_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()

            avg_train_loss += loss / len(train_loader)

        # выводим средний лосс на тренировочной выборке за эпоху
        print('avg train loss: %f' % avg_train_loss)

        # 2. Тестируем сеть на картинках из val_loader
        model.eval()

        avg_val_loss = 0
        for i, X_batch in enumerate(val_loader):
            # переносим батч на GPU
            X_batch = X_batch.to(device)
            # получаем ответы сети на батч
            Y_pred = model(X_batch)
            # считаем лосс на батче
            loss = loss_fn(Y_pred, X_batch)

            avg_val_loss += loss / len(val_loader)

        # выводим средний лосс на валидационных данных
        print('avg val loss: %f' % avg_val_loss)

        # 3. Визуализируем ответы сети на шести картинках из валидационных данных

        # получаем один батч из data_val
        X_val = next(iter(val_loader))
        # получаем ответ сети на картинки из батча
        Y_pred = model(X_val.to(device))
        Y_hat = Y_pred.detach().cpu().numpy()
        Y_hat = np.argmax(Y_hat, axis=1)

        # удаляем предыдущую визуализацию
        clear_output(wait=True)

        _, axes = plt.subplots(2, 6, figsize=(6 * 4, 2 * 4))
        for k in range(6):
            # отрисовываем 6 картинок, поданных на вход сети
            # картинки нормализованы, поэтому могут выглядеть непривычно
            axes[0][k].imshow(denorm(X_val[k].data.cpu().numpy()).transpose(1, 2, 0), aspect='auto')
            axes[0][k].title.set_text('Input')

            # отрисовываем ответы сети для каждого из четырех классов сегментации в отдельности
            axes[1][k].imshow(denorm(Y_pred[k].data.cpu().numpy()).transpose(1, 2, 0), aspect='auto')
            axes[1][k].title.set_text('Output')
        plt.suptitle('%d / %d - val loss: %f' % (epoch + 1, epochs, avg_val_loss))
        plt.show()


autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

train(autoencoder, optimizer, criterion, 10, train_loader, val_loader)