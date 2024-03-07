import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

from denoiser.data import  process

def out2rgb(out, wb, cam2rgb):
    rgb_out_torch = process(out, wb, cam2rgb, gamma=2.2)
    return rgb_out_torch


def out2rgb_calculate_score(out, gt, wb, cam2rgb):
    """
    out: pred
    gt: ccm
    """
    ## prepare for process
    ## the process function only support batch format, which is (B, 4, H, W)
    rgb_out_torch = process(out, wb, cam2rgb, gamma=2.2)
    rgb_out = rgb_out_torch.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    rgb_gt_torch = process(gt, wb, cam2rgb, gamma=2.2)
    rgb_gt = rgb_gt_torch.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


    def _score_base(_psnr, _ssim):
        return _psnr + np.log(_ssim) / np.log(1.2)

    # calculate metrics on rgb domain
    psnr_v = psnr(rgb_gt, rgb_out)
    ssim_v = ssim(rgb2gray(rgb_gt), rgb2gray(rgb_out), data_range=1)
    score = _score_base(psnr_v, ssim_v)

    return score, psnr_v, ssim_v, rgb_out_torch, rgb_gt_torch




def calculate_score(out, gt):
    def _score_base(_psnr, _ssim):
        return _psnr + np.log(_ssim) / np.log(1.2)

    # calculate metrics on rgb domain
    psnr_v = psnr(gt, out)
    ssim_v = ssim(rgb2gray(gt), rgb2gray(out))
    score = _score_base(psnr_v, ssim_v)

    return score, psnr_v, ssim_v
