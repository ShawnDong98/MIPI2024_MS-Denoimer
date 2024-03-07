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



dataset.train.opt.crop_size = 960
dataset.train.opt.load_in_mem = True

dataset.valid.opt.load_in_mem = True

dataset.test_a.opt.dataroot = ['./datasets/RawDenoising/test/Camera1/short/', './datasets/RawDenoising/test/Camera2/short/']



model = OmegaConf.create()

model.arch = L(MultiStageLNLT)(
    stages = 5,
    num_level = 4,
    share_params = False,
    in_dim = 4,
    dim = 28,
    out_dim = 4,
    window_sizes = [(16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16)],
    layernorm_type = "BiasFree",
    ffn_expansion_factor = 2.66,
    num_blocks = [1, 1, 2, 3, 2, 1, 1],
    losses = {
        "l1_loss": L(CharbonnierLoss)(weight=1),
        "tv_loss": L(TVLoss)(weight=0.1),
        "lap_loss": L(LapLoss)(
            weight = 0.1,
            max_levels = 3,
            channels = 4,
        )
    },
    pretrain=False,
)

model.ema = L(ExponentialMovingAverage)(
    parameters=[],
    decay=0.999
)

model.autocast = True 


optimizer.lr = 3e-5


train.epochs = 100

train.debug = True
train.output_dir = f"./exp/ms_lnlt_Sony_Pretrain_levelx{model.arch.num_level}_block{''.join([str(i) for i in model.arch.num_blocks])}_{model.arch.stages}stg_patch960/"
train.resume_ckpt_path = f""



test.pretrained_ckpt_path = f"./exp/ms_lnlt_SonyNikon_TV_Pretrain_finetune_levelx{model.arch.num_level}_block{''.join([str(i) for i in model.arch.num_blocks])}_{model.arch.stages}stg_patch960/2024_03_03_15_06_44/model_epoch_299.pth"

test.output_dir = f"./exp/ms_lnlt_SonyNikon_TV_Pretrain_finetune_NoTV_levelx{model.arch.num_level}_block{''.join([str(i) for i in model.arch.num_blocks])}_{model.arch.stages}stg_patch960/2024_03_03_17_55_39/"
