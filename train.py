import argparse
import collections
import warnings

import numpy as np
import torch

from src.model import DCGAN
from src.trainer import Trainer
from src.dataset import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main():
    # setup dataloaders here 
    dataloaders = get_dataloaders("data/cats")

    # build model architecture
    HIDDEN_DIM = 100
    FMAP_DIM = 64
    N_CHANNELS = 3

    model = DCGAN(
        hidden_dim=HIDDEN_DIM,
        feature_map_dim=FMAP_DIM,
        n_channels=N_CHANNELS
    )
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of parameters in the generator: {sum(p.numel() for p in model.generator.parameters())}")
    print(f"Number of parameters in the discriminator: {sum(p.numel() for p in model.discriminator.parameters())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # get function handles of loss and metrics
    generator_loss_module = torch.nn.BCELoss()
    discriminator_loss_module = torch.nn.BCELoss()

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    generator_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    discriminator_trainable_params = filter(lambda p: p.requires_grad, model.discriminator.parameters())
    generator_optimizer = torch.optim.Adam(generator_trainable_params, lr=3e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator_trainable_params, lr=3e-4, betas=(0.5, 0.999))
    generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.99)
    discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.99)

    # TODO

    


if __name__ == "__main__":
    main()