from omegaconf import OmegaConf

from torch_ema import ExponentialMovingAverage

from denoiser.data import *
from denoiser.config import LazyCall as L
from denoiser.architectures import *
from denoiser.losses import *

from .common.data import dataset
from .common.optim import AdamW as optimizer
from .common.schedule import scheduler
from .common.train import train
from .common.test import test




dataset.train.opt.dataroot = [
                                './datasets/RawDenoising/train/Camera1/', 
                                './datasets/RawDenoising/train/Camera2/',
                                './datasets/RawDenoising/SID/Sony/',
                            ]

dataset.train.opt.crop_size = 960
dataset.train.opt.load_in_mem = True

dataset.valid.opt.load_in_mem = True


model = OmegaConf.create()

model.arch = L(MultiStageDenoimerProfiling)(
    stages = 3,
    num_level = 4,
    share_params = False,
    in_dim = 4,
    dim = 28,
    out_dim = 4,
    window_sizes = [(16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16)],
    layernorm_type = "BiasFree",
    ffn_expansion_factor = 2.66,
    num_blocks = [1, 1, 3, 4, 3, 1, 1],
    losses = {
        "l1_loss": L(CharbonnierLoss)(weight=1),
        "tv_loss": L(TVLoss)(weight=0.1),
        "lap_loss": L(LapLoss)(
            weight = 0.1,
            max_levels = 3,
            channels = 4,
        )
    },
    pretrain=True,
)

model.ema = L(ExponentialMovingAverage)(
    parameters=[],
    decay=0.999
)

model.autocast = True 


optimizer.lr = 3e-4


train.epochs = 300

train.debug = True
train.output_dir = f"./exp/ms_denoimer_pretrain_levelx{model.arch.num_level}_block{''.join([str(i) for i in model.arch.num_blocks])}_{model.arch.stages}stg_patch960/"
train.resume_ckpt_path = f""


test.pretrained_ckpt_path = f""

test.output_dir = f"./results/"
