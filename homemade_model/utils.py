import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path):
    """Save first image in list"""
    images = [Image.fromarray(img) for img in images]
    images[0].save(path)



def get_data(args):
    """Random horizontal flip, to tensor, normalize"""
    train_transforms = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Mean for dataset 2
        transforms.Normalize(mean=[0.376, 0.360, 0.331], std=[0.322, 0.316, 0.309]),
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

