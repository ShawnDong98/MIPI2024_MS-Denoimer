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
    pretrain=False,
)

model.ema = L(ExponentialMovingAverage)(
    parameters=[],
    decay=0.999
)

model.autocast = True 


# optimizer.lr = 3e-4 if model.arch.pretrain else 1e-4
optimizer.lr = 1e-4


train.epochs = 200

train.debug = True
train.output_dir = f"./exp/ms_denoimer_SonyNikon_TV_Pretrain_finetune_NoTV_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/"
train.resume_ckpt_path = f""


# test.pretrained_ckpt_path = f"./exp/ms_denoimer_Sony_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch{dataset.train.opt.crop_size}/2024_02_22_21_21_15/model_epoch_290.pth"



test.pretrained_ckpt_path = [
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.archs.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_150.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_160.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_165.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_170.pth",
    f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_175.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_185.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_195.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_206.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_215.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_225.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_235.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_245.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_255.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_266.pth",
    # f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/model_epoch_276.pth",
    # f"./exp/ms_denoimer_Pretrain_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_03_02_11_05_46/model_epoch_199.pth", 
    # f"./exp/ms_denoimer_SonyNikon_TV_Pretrain_finetune_NoTV_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_03_03_18_53_39/model_epoch_198.pth"
]


test.swa_weights = [1.0]



# test.pretrained_ckpt_path = [f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_03_03_18_53_39/model_epoch_198.pth"]

test.output_dir = f"./exp/ms_denoimer_Sony_finetune_levelx{model.arch.num_level}_block1134311_{model.arch.stages}stg_patch960/2024_02_25_14_28_02/"
