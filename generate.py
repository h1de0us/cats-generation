import torch
import torchvision
import os

from src.model import DCGAN

CLIP_VALUE = 70
HIDDEN_DIM = 100
FMAP_DIM = 64
N_CHANNELS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed_noise = torch.randn(64, HIDDEN_DIM, 1, 1, device=device)

model = DCGAN(
    hidden_dim=HIDDEN_DIM,
    feature_map_dim=FMAP_DIM,
    n_channels=N_CHANNELS
)

# hardcoded value but......
path = "saved/checkpoint-epoch-clipped50-grad-epoch-70.pth"
print("Loading checkpoint: {} ...".format(path))
checkpoint = torch.load(path, map_location=device)
state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict)

os.makedirs("results", exist_ok=True)
with torch.no_grad():
    fake = model(fixed_noise).detach().cpu()
    image = torchvision.utils.make_grid(fake, padding=2, normalize=True)
    torchvision.utils.save_image(image, "results/test_image.png")



