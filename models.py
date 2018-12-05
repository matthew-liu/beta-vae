import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

# CelebA (VAE)
# Input 64x64x3.
# Adam 1e-4
# Encoder Conv 32x4x4 (stride 2), 32x4x4 (stride 2), 64x4x4 (stride 2),
# 64x4x4 (stride 2), FC 256. ReLU activation.
# Latents 32
# Decoder Deconv reverse of encoder. ReLU activation. Gaussian.

# Test set: Average loss: 2.4985, Accuracy: 4061/10000 (41%)

Base_Accuracy = 0
d_out = 200  # number of classes


# input size: 64, 64, 3

class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()

        self.best_accuracy = Base_Accuracy

        # encoder
        self.encoder = nn.Sequential(
            self._conv(3, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )

    def forward(self, x):
        x = self.encoder(x)
        print(x.size())
        exit()
        return x

    # layers
    @staticmethod
    def _conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    @staticmethod
    def loss(recon_x, x, mu, logvar, beta):
        # Reconstruction + KL divergence losses summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl_diverge

    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        # save the model if it is the best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            utils.save(self, file_path, num_to_keep)
            print("new best model with accuracy", str(accuracy), "saved!")

    def load_model(self, file_path):
        utils.restore(self, file_path)

    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)
