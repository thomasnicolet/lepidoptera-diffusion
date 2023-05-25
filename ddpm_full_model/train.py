import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch import optim
import argparse

from utils import get_data
from utils import setup_logging
from utils import save_images

from unet import UNetConditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler


def train(args):

    # Setup logging
    setup_logging(args.run_name)
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    # Setup data
    dataloader = get_data(args)

    # Init UNetConditional model
    model = UNetConditional(c_in=args.latent_ch,
                            c_out=args.latent_ch,
                            num_classes=args.num_classes,
                            image_size=args.latent_img_size,
                            device=args.device).to(args.device)

    # Init EMA model
    ema = EMA(beta=args.beta)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # Load pretrained VAE
    vae = AutoencoderKL(in_channels=args.img_ch, out_channels=args.img_ch, latent_channels=args.latent_ch)
    vae = vae.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(args.device)

    # Init optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Init loss function
    mse = nn.MSELoss()

    # Init custom diffusion class for forward/backward process
    diffusion = Diffusion(img_size=args.latent_img_size, channel_size=args.latent_ch, device=args.device)

    tepochs = trange(args.epochs)

    # Hard coded butterfly label to sample
    butterfly_label_str = "Lepidoptera Arctiidae Thyrosticta" # Dataset 1
    #butterfly_label_str = "Lepidoptera Papilionidae Parnassius" # Dataset 2
    butterfly_label_idx = dataloader.dataset.class_to_idx[butterfly_label_str]
    butterfly_label = torch.tensor([butterfly_label_idx]).to(args.device)

    for epoch in tepochs:
        logging.info(f"Starting epoch {epoch + 1} of {args.epochs}")

        for i, (images, labels) in enumerate(tqdm(dataloader, leave=False)):

            # Send data to device(cuda)
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Sample timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)

            # Make latent distribution from vae
            latent_dist = vae.encode(images).latent_dist.sample()
            latent_dist = latent_dist * vae.config.scaling_factor

            # Make noise through diffusion process, and save noise as target
            x_t, noise = diffusion.noise_images(latent_dist, t)

            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(x_t, t, labels)

            # Calculate loss
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA model
            ema.step_ema(ema_model, model)

            # Update tqdm logging
            tepochs.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(),
                              global_step=epoch * len(dataloader) + i)

        save_every_n_epoch = 1
        if epoch % save_every_n_epoch == 0:

            # Hard coded butterfly label
            ema_latents = diffusion.sample_scheduler(ema_model, args.scheduler, n=1, labels=butterfly_label)

            # Scale latents
            ema_latents = 1 / vae.config.scaling_factor * ema_latents

            # Decode latents
            with torch.no_grad():
                ema_sampled_images = vae.decode(ema_latents).sample

            # Scale EMA images
            ema_sampled_images = (ema_sampled_images / 2 + 0.5).clamp(0, 1)
            ema_sampled_images = ema_sampled_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            ema_sampled_images = (ema_sampled_images * 255).round().astype("uint8")

            # Save images
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

            # Save models
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

    print("Done running!")

    logger.close()


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "ddpm_full_model"
    args.epochs = 50
    args.batch_size = 1
    args.img_ch = 3
    args.latent_ch = 4
    args.latent_img_size = 64
    args.img_size = 512
    args.beta = 0.995  # For EMA
    args.num_classes = 9
    args.dataset_path = "~/datasets/dataset2/train_conditional"
    args.device = "cuda"
    args.lr = 1e-4
    args.weight_decay = 0.1
    args.scheduler = DDPMScheduler(num_train_timesteps=1000)

    train(args)


if __name__ == "__main__":
    launch()
