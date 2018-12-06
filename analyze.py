import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import numpy as np
import models
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image


# generate n=num images using the model
def generate(model, num, device):
    model.eval()
    z = torch.randn(num, model.latent_size).to(device)
    with torch.no_grad():
        return model.decode(z).cpu()


def linear_interpolate(im1, im2, model, device):
    model.eval()
    im1, im2 = torch.unsqueeze(im1, dim=0), torch.unsqueeze(im2, dim=0)
    im1, im2 = im1.to(device), im2.to(device)

    factors = np.linspace(1, 0, num=10)
    result = []

    with torch.no_grad():
        mu1, logvar1 = model.encode(im1)
        z1 = model.sample(mu1, logvar1)

        mu2, logvar2 = model.encode(im2)
        z2 = model.sample(mu2, logvar2)

        for f in factors:
            z = (f * z1 + (1 - f) * z2).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)

    return result


def plot_loss(train_loss, test_loss, filepath):
    train_x, train_l = zip(*train_loss)
    test_x, test_l = zip(*test_loss)
    plt.figure()
    plt.title('Train Loss vs. Test Loss')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.plot(train_x, train_l, 'b', label='train_loss')
    plt.plot(test_x, test_l, 'r', label='test_loss')
    plt.legend()
    plt.savefig(filepath)


def get_attr_ims(attr, num=10):
    ids = prep.get_attr(attr_map, id_attr_map, attr)
    dataset = prep.ImageDiskLoader(ids)
    indices = np.random.randint(0, len(dataset), num)
    ims = [dataset[i] for i in indices]
    idx_ids = [dataset.im_ids[i] for i in indices]
    return ims, idx_ids


USE_CUDA = True
MODEL = 'l32-e100'
MODEL_PATH = './checkpoints/' + MODEL
LOG_PATH = './logs/' + MODEL + '/log.pkl'
OUTPUT_PATH = './samples/'
PLOT_PATH = './plots/' + MODEL
LATENT_SIZE = 32

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
print('latent size:', model.latent_size)

attr_map, id_attr_map = prep.get_attributes()

if __name__ == "__main__":

    # model.load_last_model(MODEL_PATH)

    # samples = generate(model, 60, device)
    # save_image(samples, OUTPUT_PATH + MODEL + '.png', padding=0, nrow=10)
    #
    # train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))
    # plot_loss(train_losses, test_losses, PLOT_PATH)

    ims, im_ids = get_attr_ims('smiling', num=10)
    utils.show_images(ims, tensor=True)
    print(im_ids)
    # inter_ims = linear_interpolate(ims[0], ims[1], model, device)

    # save_image(inter_ims, OUTPUT_PATH + 'interpolate' + '.png', padding=0, nrow=10)
