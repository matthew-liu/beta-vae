from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import preprocess as prep
import models


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

USE_CUDA = True
PRINT_INTERVAL = 100

# training code
train_ids, test_ids = prep.split_dataset()
print('num train_images:', len(train_ids))
print('num test_images:', len(test_ids))

data_train = prep.ImageLoader(train_ids)
data_test = prep.ImageLoader(test_ids)

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

# attr_map, id_attr_map = prep.get_attributes()
# smile_ims = prep.get_attr(attr_map, id_attr_map, 'smiling')

model = models.BetaVAE()
# print(model)

for batch_idx, data in enumerate(train_loader):
    output = model(data)