import warnings

from src.model import DCGAN

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
import numpy as np

import wandb
import os 

from torch.nn.utils import clip_grad_norm_

from piq import FID, ssim

from torch.utils.data import DataLoader
from src.dataset import get_dataloaders, collate_fn

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)



@torch.no_grad()
def get_grad_norm(model, norm_type=2, type: str = "generator"):
    if type == "generator":
        parameters = model.generator.parameters()
    elif type == "discriminator":
        parameters = model.discriminator.parameters()
    else:
        raise NotImplementedError
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()


def main():
    # setup dataloaders here 
    dataloaders = get_dataloaders("data/cats/cats")

    CLIP_VALUE = 70
    HIDDEN_DIM = 100
    FMAP_DIM = 64
    N_CHANNELS = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fixed_noise = torch.randn(64, HIDDEN_DIM, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    transform = transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

    dataloader = get_dataloaders('data/cats/cats',
                                transform=transform,
                                batch_size=64)["train"]
    

    os.environ["WANDB_API_KEY"] = "<YOUR_API_KEY>"
    wandb.login()


    model = DCGAN(
        hidden_dim=HIDDEN_DIM,
        feature_map_dim=FMAP_DIM,
        n_channels=N_CHANNELS
    )

    model = model.to(device)

    # get function handles of loss and metrics
    generator_loss = torch.nn.BCELoss()
    discriminator_loss = torch.nn.BCELoss()

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    generator_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    discriminator_trainable_params = filter(lambda p: p.requires_grad, model.discriminator.parameters())
    generator_optimizer = torch.optim.Adam(generator_trainable_params, lr=3e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator_trainable_params, lr=3e-4, betas=(0.5, 0.999))
    generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.99)
    discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.99)

    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of parameters in the generator: {sum(p.numel() for p in model.generator.parameters())}")
    print(f"Number of parameters in the discriminator: {sum(p.numel() for p in model.discriminator.parameters())}")


    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    num_epochs = 70

    wandb.init()

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, batch in enumerate(dataloader, 0):
            data = batch["images"]
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator_optimizer.zero_grad()
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = model.discriminator(real_cpu).view(-1)
            error_disc_real = discriminator_loss(output, label)
            error_disc_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(batch_size, HIDDEN_DIM, 1, 1, device=device)
            fake = model(noise)
            label.fill_(fake_label)
            output = model.discriminator(fake.detach()).view(-1)
            error_disc_fake = discriminator_loss(output, label)
            error_disc_fake.backward()
            D_G_z1 = output.mean().item()

            error_disc = error_disc_real + error_disc_fake
            discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator_optimizer.zero_grad()
            label.fill_(real_label)  
            output = model.discriminator(fake).view(-1)
            error_gen = generator_loss(output, label)
            error_gen.backward()
            D_G_z2 = output.mean().item()
            generator_optimizer.step()
            
            clip_grad_norm_(model.parameters(), CLIP_VALUE)

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        error_disc.item(), error_gen.item(), D_x, D_G_z1, D_G_z2))
                wandb.log({
                    "discriminator loss": error_disc.item(),
                })
                wandb.log({
                    "generator loss": error_gen.item(),
                })
                wandb.log({
                    "generator grad norm": get_grad_norm(model, type="generator")
                })
                wandb.log({
                    "discriminator grad norm": get_grad_norm(model, type="discriminator")
                })
                

            # Save Losses for plotting later
            G_losses.append(error_gen.item())
            D_losses.append(error_disc.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = model(fixed_noise).detach().cpu()
                image = torchvision.utils.make_grid(fake, padding=2, normalize=True)
                img_list.append(image)
                
                wandb.log({
                    "generated images": wandb.Image(image)})
                
                # calculate metrics
                # scale data from 0 to 1
                if fake.shape != data.shape:
                    continue

                fake = fake * 0.5 + 0.5
                data = data * 0.5 + 0.5
                ssim_index = ssim(fake, data, data_range=1.) # from 0 to 1
                
                fake_loader = DataLoader(fake, collate_fn=collate_fn)
                real_loader = DataLoader(data, collate_fn=collate_fn)
                fid_metric = FID()
                fake_feats = fid_metric.compute_feats(fake_loader)
                real_feats = fid_metric.compute_feats(real_loader)
                fid = fid_metric(fake_feats, real_feats)
                
                wandb.log({
                    "ssim": ssim_index.mean()})
                
                wandb.log({
                    "fid": fid.mean()})

            iters += 1
            
    wandb.finish()

    os.makedirs("saved", exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
    }
    filename = f"saved/checkpoint-epoch-clipped{CLIP_VALUE}-grad-epoch-{epoch + 1}.pth"
    torch.save(state, filename)

    # gif creation, works only in the notebook
    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())
    


if __name__ == "__main__":
    main()