import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch import optim
import argparse

from utils import get_data
from utils import setup_logging
from utils import save_images

from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(image_size=args.img_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dataloader_length = len(dataloader)  # TODO: Update name

    # Hack
    # torch.backends.cudnn.enabled = False 

    tepochs = trange(args.epochs)

    for epoch in tepochs:
        # TODO: How does this work?
        logging.info(f"Starting epoch {epoch + 1} of {args.epochs}")

        for i, (images, test) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)  # Forward process
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepochs.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(),
                              global_step=epoch * dataloader_length + i)

        # TODO: For real tests, we want to print sample images every epoch
        if epoch % 10 == 0:
            # TODO: This is ([1, 3, 64, 64]). Is n ever NOT 1?
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join(
                "results", args.run_name, f"epoch_{epoch}.png"))
            torch.save(model.state_dict(), os.path.join(
                "models", args.run_name, f"ckpt.pt"))
    else:
        print("Done running!")

    # TODO: What does close do on SummaryWriter?
    logger.close()


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "ddpm64_full"
    args.epochs = 500
    args.batch_size = 2
    args.img_size = 64
    args.dataset_path = "~/datasets/resized_64"  # "../../data/lepicropd/resized_32"
    args.device = "cuda"
    args.lr = 1e-4
    args.weight_decay = 0.1

    train(args)


if __name__ == "__main__":
    launch()
