import torch
import torch.optim as optim
import numpy as np
import multiprocessing
import time
import preprocess as prep
import models
import utils


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(np.transpose(data[0].cpu(), (1, 2, 0)))
                rect_images.append(np.transpose(output[0].cpu(), (1, 2, 0)))

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss


# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 300

LEARNING_RATE = 1e-4

USE_CUDA = True
PRINT_INTERVAL = 100
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './comparisons/'

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

# training code
train_ids, test_ids = prep.split_dataset()
print('num train_images:', len(train_ids))
print('num test_images:', len(test_ids))


data_train = prep.ImageMemoryLoader(train_ids)
data_test = prep.ImageMemoryLoader(test_ids)

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

model = models.BetaVAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_epoch = model.load_last_model(MODEL_PATH) + 1

train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))

for epoch in range(start_epoch, EPOCHS + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
    test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

    utils.compare_2_image_arrays(original_images, rect_images, path=COMPARE_PATH + str(epoch))

    train_losses.append((epoch, train_loss))
    test_losses.append((epoch, test_loss))
    utils.write_log(LOG_PATH, (train_losses, test_losses))

    model.save_model(MODEL_PATH + '%03d.pt' % epoch)
