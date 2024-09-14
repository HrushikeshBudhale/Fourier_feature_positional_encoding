import os
import torch
import imageio
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_gif(image_folder:str, gif_name:str, duration:float=5.0):
    frames = [imageio.imread(image_folder + image_name) for image_name in sorted(os.listdir(image_folder))]
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1, img2 = img1.detach().clone(), img2.detach().clone()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return (20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))).item()

