from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.functional import crop
import numpy as np
import matplotlib.pyplot as plt
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


train_ids, test_ids = prep.split_dataset()
print('#train_images:', len(train_ids))
print('#test_images:', len(test_ids))

# attr_map, id_attr_map = prep.get_attributes()
# smile_ims = prep.get_attr(attr_map, id_attr_map, 'smiling')

data_train = prep.ImageLoader(train_ids)
data_test = prep.ImageLoader(test_ids)

model = models.BetaVAE()
print(model)