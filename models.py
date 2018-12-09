import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# CelebA (VAE)
# Input 64x64x3.
# Adam 1e-4
# Encoder Conv 32x4x4 (stride 2), 32x4x4 (stride 2), 64x4x4 (stride 2),
# 64x4x4 (stride 2), FC 256. ReLU activation.
# Latents 32
# Decoder Deconv reverse of encoder. ReLU activation. Gaussian.


class BetaVAE(nn.Module):

    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            self._conv(3, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 3),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 64, 2, 2)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size

    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        utils.restore(self, file_path)

    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)


class DFCVAE(BetaVAE):

    def __init__(self, latent_size=100, beta=1):
        super(DFCVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.e1 = self._conv(3, 32)
        self.e2 = self._conv(32, 64)
        self.e3 = self._conv(64, 128)
        self.e4 = self._conv(128, 256)
        self.fc_mu = nn.Linear(4096, latent_size)
        self.fc_var = nn.Linear(4096, latent_size)

        # decoder
        self.d1 = self._upconv(256, 128)
        self.d2 = self._upconv(128, 64)
        self.d3 = self._upconv(64, 32)
        self.d4 = self._upconv(32, 3)
        self.fc_z = nn.Linear(latent_size, 4096)

    def encode(self, x):
        x = F.leaky_relu(self.e1(x))
        x = F.leaky_relu(self.e2(x))
        x = F.leaky_relu(self.e3(x))
        x = F.leaky_relu(self.e4(x))
        x = x.view(-1, 4096)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 256, 4, 4)
        z = F.leaky_relu(self.d1(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d2(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d3(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d4(F.interpolate(z, scale_factor=2)))
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )