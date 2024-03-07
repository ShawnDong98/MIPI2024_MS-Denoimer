from omegaconf import OmegaConf

from denoiser.config import LazyCall as L

## The dataset can be also found in `led/data/mipi_dataset.py`
## In this way you could use our codebase for both training and evaluation.
from denoiser.data import MIPIDataset, MIPITestDataset, depack_meta, calculate_ratio, process, VirtualNoisyPairGenerator

dataset = OmegaConf.create()

dataset.train = L(MIPIDataset)(
    opt={
            ## Basic options of the dataset
            'phase': 'train',                           # the phase: train or valid
            'dataroot': [
                './datasets/RawDenoising/train/Camera1/', 
                './datasets/RawDenoising/train/Camera2/',
                # './datasets/RawDenoising/SID/Sony/',
                # './datasets/RawDenoising/ELD_npy/SonyA7S2/',
                # './datasets/RawDenoising/ELD_npy/NikonD850/',
                # './datasets/RawDenoising/ELD_npy/CanonEOS70D/',
                # './datasets/RawDenoising/ELD_npy/CanonEOS700D/'
            ],         # the path to our released dataset
            # 'dataroot': ['./datasets/RawDenoising/train/Camera1/'],         # the path to our released dataset
            'data_pair_list': [
                './datasets/RawDenoising/train/Camera1/train.txt', 
                './datasets/RawDenoising/train/Camera2/train.txt',
                # './datasets/RawDenoising/SID/Sony/train.txt',
                # './datasets/RawDenoising/ELD_npy/SonyA7S2/SonyA7S2_list.txt',
                # './datasets/RawDenoising/ELD_npy/NikonD850/NikonD850_list.txt',
                # './datasets/RawDenoising/ELD_npy/CanonEOS70D/CanonEOS70D_list.txt',
                # './datasets/RawDenoising/ELD_npy/CanonEOS700D/CanonEOS700D_list.txt',
            ], # the path to the txt file of the mipi dataset
            ## Augmentation options
            'crop_size': 256,                          # whether to crop the paired data (center crop when not in train phase)
            'use_hflip': True,                          # whether to use the flip augmentation
            'use_rot':   True,                          # whether to use the rotation augmentation
            'load_in_mem' : True,
            'batch_size' : 1,
            'num_workers' : 0,
        },
)

dataset.valid = L(MIPIDataset)(opt={
    ## Basic options of the dataset
    'phase': 'valid',                           # the phase: train or valid
    'dataroot': ['./datasets/RawDenoising/train/Camera1/', './datasets/RawDenoising/train/Camera2/'],         # the path to our released dataset
    'data_pair_list': ['./datasets/RawDenoising/train/Camera1/train.txt', './datasets/RawDenoising/train/Camera2/train.txt'], # the path to the txt file of the mipi dataset
    ## Augmentation options
    'crop_size': 960,                          # whether to crop the paired data (center crop when not in train phase)
    'use_hflip': True,                          # whether to use the flip augmentation
    'use_rot':   True,                          # whether to use the rotation augmentation
    'load_in_mem' : True,
    'batch_size' : 1,
    'num_workers' : 0,
})

dataset.test_a = L(MIPITestDataset)(opt={
    ## Basic options of the dataset
    'phase': 'test',                           # the phase: train or valid
    'dataroot': ['./datasets/RawDenoising/valid/Camera1/short/', './datasets/RawDenoising/valid/Camera2/short/'],         # the path to our released dataset
    'load_in_mem' : True,
    'batch_size' : 1,
    'num_workers' : 0,
})