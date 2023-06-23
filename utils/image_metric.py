import torch
import numpy as np


def PSNR(img_ture, img_pred):
    diff = img_pred - img_ture
    mse = torch.mean(torch.square(diff))
    return 10 * torch.log10(1 / mse)


def npPSNR(img_ture, img_pred):
    diff = np.float32(img_pred) - np.float32(img_ture)
    mse = np.mean(np.square(diff))
    return 10 * np.log10(255 * 255 / mse)
