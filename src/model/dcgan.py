import torch 
from torch import nn


# TODO: some GAN hacks?

class Generator(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 feature_map_dim,
                 n_channels):
        super().__init__()

        self.layers = nn.Sequential(
            # trying to map latent vector z to data-space
            nn.ConvTranspose2d(hidden_dim, feature_map_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_dim * 8),
            nn.ReLU(True),
            # (feature_map_dim * 8) x 4 x 4
            nn.ConvTranspose2d(feature_map_dim * 8, feature_map_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 4),
            nn.ReLU(True),
            # (feature_map_dim*4) x 8 x 8
            nn.ConvTranspose2d( feature_map_dim * 4, feature_map_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 2),
            nn.ReLU(True),
            # (feature_map_dim * 2) x 16 x 16
            nn.ConvTranspose2d( feature_map_dim * 2, feature_map_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim),
            nn.ReLU(True),
            # (feature_map_dim) x 32 x 32
            nn.ConvTranspose2d( feature_map_dim, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (n_channels) x 64 x 64
        )
        self.initialize_weights()

    def forward(self, input):
        return self.layers(input)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    


class Discriminator(nn.Module):
    def __init__(self, 
                 feature_map_dim,
                 n_channels,):
        super().__init__()
        self.main = nn.Sequential(
            # input is ``(n_channels) x 64 x 64``
            nn.Conv2d(n_channels, feature_map_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_map_dim) x 32 x 32``
            nn.Conv2d(feature_map_dim, feature_map_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_map_dim * 2) x 16 x 16``
            nn.Conv2d(feature_map_dim * 2, feature_map_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_map_dim * 4) x 8 x 8``
            nn.Conv2d(feature_map_dim * 4, feature_map_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_map_dim*8) x 4 x 4``
            nn.Conv2d(feature_map_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class DCGAN(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 feature_map_dim,
                 n_channels):
        super().__init__()
        self.generator = Generator(hidden_dim, feature_map_dim, n_channels)
        self.discriminator = Discriminator(feature_map_dim, n_channels)

    def forward(self, input):
        return self.generator(input)
    
    def initialize_weights(self):
        self.generator.initialize_weights()
        self.discriminator.initialize_weights()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))