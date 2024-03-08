import os
import sys
import time
from omegaconf import OmegaConf

from collections import Sequence

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
## save the image
from torchvision.utils import save_image

from matplotlib import pyplot as plt

from glob import glob
from tqdm import tqdm
import numpy as np


from denoiser.engine import default_argument_parser, default_setup
from denoiser.config import LazyConfig, instantiate

## The dataset can be also found in `led/data/mipi_dataset.py`
## In this way you could use our codebase for both training and evaluation.
from denoiser.data import MIPIDataset, depack_meta, calculate_ratio, process

from denoiser.metrics import out2rgb, out2rgb_calculate_score
from denoiser.utils.utils import checkpoint
from denoiser.utils.swa import apply_swa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configs
args = default_argument_parser().parse_args()
cfg = LazyConfig.load(args.config_file)
cfg.DETERMINISTIC = True

test_a_dataset = instantiate(cfg.dataset.test_a)

# Models
models = [instantiate(arch).to(device) for arch in cfg.model.archs]



if cfg.test.pretrained_ckpt_path:
    print(f"===> Loading Checkpoint from {cfg.test.pretrained_ckpt_path}")
    for model, ckpt, swa_weight in zip(models, cfg.test.pretrained_ckpt_path, cfg.test.swa_weights):
        if isinstance(cfg.test.pretrained_ckpt_path, Sequence):
            print("SWA....")
            model = apply_swa(model, ckpt, swa_weight, device=device)
        else:
            save_state = torch.load(ckpt, map_location=device)
            model.load_state_dict(save_state['model'])



def test(test_loader):
    begin = time.time()
    model.eval()
    train_tqdm = tqdm(enumerate(test_loader), total=len(test_a_dataset))
    for batch_idx, data in train_tqdm:
        lq_path = data['lq_path'][0]
        data = {k:v.to(device) for k, v in data.items() if type(v) is torch.Tensor}
        with torch.no_grad():
            out = model.forward_test_tta(data)



        rgb_out_torch = out2rgb(out, data['wb'], data['ccm'])

         ## save image
        path_splits = lq_path.split('/')
        curr_cam = path_splits[-3]
        curr_im_name = path_splits[-1].replace('.npz', '.png')
        if not os.path.exists(f'{cfg.test.output_dir}/test_save_img/{curr_cam}'):
            os.makedirs(f'{cfg.test.output_dir}/test_save_img/{curr_cam}')
            save_image(rgb_out_torch, f'{cfg.test.output_dir}/test_save_img/{curr_cam}/{curr_im_name}')
        else:
            save_image(rgb_out_torch, f'{cfg.test.output_dir}/test_save_img/{curr_cam}/{curr_im_name}')
    
    end = time.time()

    
    print('===> testing time: {:.2f}'
                .format((end - begin)))
    
    return out


def main():
    test_loader = DataLoader(
            dataset = test_a_dataset,
            batch_size = cfg.dataset.test_a.opt.batch_size,
            shuffle = False,
            num_workers = cfg.dataset.test_a.opt.num_workers,
            pin_memory = False,
            drop_last = False
        )
    out = test(test_loader)
if __name__ == "__main__":
    main()
