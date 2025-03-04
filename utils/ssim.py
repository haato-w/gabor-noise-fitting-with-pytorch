"""
I borrowed these loss functions from https://github.com/OutofAi/2D-Gaussian-Splatting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_window(window_size: int, channel: int) -> torch.Tensor:
    def gaussian(window_size, sigma):
        gauss = torch.exp(
            torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma * 2) 
                for x in range(window_size)]))
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11, size_average=True) -> torch.Tensor:
    (_, _, channel) = img1.size()
    # Assuming the image is of shape [N, C, H, W]
    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)

    # Parameters for SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    SSIM_numerator = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    SSIM_denominator = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    SSIM = SSIM_numerator / SSIM_denominator
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def d_ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size=11, size_average=True) -> torch.Tensor:
    return ssim(img1, img2, window_size, size_average).mean()

def combined_loss(pred: torch.Tensor, target: torch.Tensor, lambda_param=0.5) -> torch.Tensor:
    l1loss = nn.L1Loss()
    return (1 - lambda_param) * l1loss(pred, target) + lambda_param * d_ssim_loss(pred, target)
