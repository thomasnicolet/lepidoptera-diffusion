import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32,32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)  # TODO: Rewrite this without double cat?
    ], dim=-2).permute(1,2,0).cpu())
    plt.show()


def save_images(image, path, **kwargs):
    grid = torchvision.utils.make_grid(image, **kwargs)
    nd_arr = grid.permute(1,2,0).cpu().numpy()
    im = Image.fromarray(nd_arr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # TODO: Consider adding num_workers
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

