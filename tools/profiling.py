import os
import sys
import time
from omegaconf import OmegaConf

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
## save the image
from torchvision.utils import save_image
from fvcore.nn import FlopCountAnalysis


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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configs
args = default_argument_parser().parse_args()
cfg = LazyConfig.load(args.config_file)
cfg.DETERMINISTIC = True

# Model
model = instantiate(cfg.model.arch).to(device)

x = torch.randn((1, 4, 960, 960)).to(device)


# with torch.autocast(device_type=str(device)):
#     flops = FlopCountAnalysis(model, (x))

# all_flops = flops.total()
# n_param = sum([p.nelement() for p in model.parameters()])
# print(f'GMac:{flops.total()/(1024*1024*1024)}')
# print(f'Params:{n_param / (1024*1024)}M')


start_time = time.time()
with torch.autocast(device_type=str(device)):
    model(x)
end_time = time.time()

print(f'Time:{end_time - start_time}s')