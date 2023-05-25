import torch
from modules import UNet
from diffusion import Diffusion
from utils import plot_images

device = "cuda"
model = UNet().to(device)
ckpt = torch.load("models/DDPM_Unconditional/ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=1)
plot_images(x)
