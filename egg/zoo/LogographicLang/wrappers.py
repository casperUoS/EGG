import torch
from torch import nn


class SenderWrapper(nn.Module):

    def __init__(self, encoder, decoder):
        super(SenderWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        return mu + std*eps, std

    def forward(self, x):
        mu, logvar, sender_aux = self.encoder(x)
        z, std = self.reparameterize(mu, logvar)
        ouput, aux = self.decoder(z)

