from omegaconf import OmegaConf


from denoiser.config import LazyCall as L
from denoiser.architectures import *

from .common.data import dataset



dataset.test_a.opt.dataroot = ['./datasets/RawDenoising/test/Camera1/short/', './datasets/RawDenoising/test/Camera2/short/']


model = OmegaConf.create()


model.archs = [
    L(MultiStageDenoimer)(
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
        losses = {},
        pretrain=False
    ),
    L(MultiStageLNLT)(
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
        losses = {},
        pretrain=False,
    ),
    L(MultiStageRestormer)(
        stages = 5,
        num_level = 4,
        share_params = False,
        in_dim = 4,
        dim = 28,
        out_dim = 4,
        layernorm_type = "BiasFree",
        ffn_expansion_factor = 2.66,
        num_blocks = [2, 2, 5, 7, 5, 2, 2],
        losses = {},
        pretrain=False,
    )
]

model.autocast = True 

test = dict(
    output_dir = f"./results/",
    pretrained_ckpt_path = [
        [
            f"./checkpoints/ms_denoimer/model_epoch_175.pth",
           
        ],
        [
            f"./checkpoints/ms_lnlt/model_epoch_299.pth"
        ],
        [
            f"./checkpoints/ms_restormer/model_epoch_199.pth"
        ]
    ],
    swa_weights = [
        [1.0],
        [1.0],
        [1.0],
    ],
    ensemble_weights = [0.35, 0.15, 0.5]
)