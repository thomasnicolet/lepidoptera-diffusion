import torch
import torch.nn as nn
from tqdm import tqdm
import logging


class Diffusion:
    """Diffusion model for image denoising.
    Args:
        noise_steps (int): Number of noise steps. T=1000 from DDPM.
        beta_start (float): Starting value for beta. 0.0004 in DDPM.
        beta_end (float): Ending value for beta. 0.02 in DDPM.
        img_size (int): Size of the image. 64x64 for computability
        device (str): Device to use. 'cuda' or 'cpu'
        """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)  # Noise schedule
        self.alpha = 1. - self.beta  # Notation from DDPM
        # Cumulative product of alpha
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """Prepares the noise schedule for the diffusion process.
        Returns:
            noise_schedule (torch.Tensor): Noise schedule for the diffusion process.
            """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x: torch.Tensor, t: int):
        """Adds noise to the images.
        Args:
            x (torch.Tensor): Images to add noise to.
            t (int): Current noise step.
        Returns:
            x (torch.Tensor): Noisy images.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[
            :, None, None, None]                # Torch broadcasting
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]  # Torch broadcasting
        noise = torch.randn_like(x)  # Sample noise
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise  # Add noise
        return x_t, noise

    def sample_timesteps(self, n: int):
        """Samples timesteps from the noise schedule.
        Args:
            n (int): Number of timesteps to sample.
        Returns:
            timesteps (torch.Tensor): Timesteps sampled from the noise schedule.
            """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model: nn.Module, n: int):
        """Samples images from the diffusion model. Follows Algorithm 2 in DDPM.
        Args:
            model (nn.Module): Model to sample from.
            n (int): Number of images to sample."""
        logging.info(f"Sampling {n} images from the diffusion model.")
        model.eval()  # Set model to eval mode
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(
                self.device)  # Sample noise
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                # Current timestep
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t)  # Predict noise

                # Torch broadcasting
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None,
                                              None, None]  # Torch broadcasting
                beta = self.beta[t][:, None, None, None]  # Torch broadcasting
                if i > 1:
                    noise = torch.randn_like(x)  # Sample noise
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                             * predicted_noise) + torch.sqrt(beta) * noise
                x = x.clamp(-1,1)

        model.train()  # Set model back to train mode
        x = (x.clamp(-1, 1) + 1) / 2  # Rescale to [0, 1]
        # Rescale to 0-255. TODO Consider .byte() to rescale
        x = (x * 255).type(torch.uint8)
        return x


if __name__ == '__main__':
    print("Ran diffusion.py")
