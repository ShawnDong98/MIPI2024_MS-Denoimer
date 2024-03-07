import os
import re
import random
from numbers import Number
from copy import deepcopy

import math
from tabulate import tabulate

import torch
from torch import nn

import numpy as np
from os import path as osp
from os.path import join
from torch.utils import data as data
from tqdm import tqdm
import pandas as pd

from glob import glob

from denoiser.config import instantiate

def calculate_ratio(lq, gt=None, camera="MIPI"):
    """
    Calculate the additional dgain which needs to add on the input.
    The Exposure value for the ground truth is default set to 3000.
    How to calculate the exposure value can be found in `docs/demo.md`
    """
    if camera == "MIPI":
        shutter_lq = float(re.search(r'_(\d+(\.\d+)?)$', os.path.splitext(lq)[0]).group(0)[1:])
        shutter_gt = float(re.search(r'_(\d+(\.\d+)?)$', os.path.splitext(gt)[0]).group(0)[1:]) \
                    if gt is not None else 3000
    if camera in ["Sony", "SonyA7S2", "NikonD850", "CanonEOS70D", "CanonEOS700D"]:
        shutter_lq = float(re.search(r'_(\d+(\.\d+)?)+s', os.path.splitext(lq)[0]).group(0)[1:-1])
        shutter_gt = float(re.search(r'_(\d+(\.\d+)?)+s', os.path.splitext(gt)[0]).group(0)[1:-1]) \
                    if gt is not None else 30

    
    return shutter_gt / shutter_lq

def depack_meta(meta, to_tensor=True):
    """ Depack the npz file and normalize the input """

    ## load meta
    if isinstance(meta, str):
        meta = np.load(meta, allow_pickle=True)

    ## read meta data
    black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
    white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
    im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
    wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
    ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
    scale = np.ascontiguousarray((meta["white_level"] - meta["black_level"]).astype('float32'))
    meta.close()

    if to_tensor:
        ## convert to tensor
        im = torch.from_numpy(im).float().contiguous()
        black_level = torch.from_numpy(black_level).float().contiguous()
        white_level = torch.from_numpy(white_level).float().contiguous()
        wb = torch.from_numpy(wb).float().contiguous()
        ccm = torch.from_numpy(ccm).float().contiguous()
        scale = torch.from_numpy(scale).float().contiguous()

    return (im - black_level) / (white_level - black_level), wb, ccm, scale, white_level, black_level


## A same dataset can be found in `led/data/mipi_dataset.py`
class MIPIDataset(data.Dataset):
    def __init__(self, opt, noise_maker=None) -> None:
        super().__init__()
        self.opt = opt

        if noise_maker: 
            self.noise_maker = instantiate(noise_maker)
        else:
            self.noise_maker = noise_maker

        self.root_folders = opt['dataroot']
        self.data_pair_lists = opt['data_pair_list']
        self.load_in_mem = opt.get('load_in_mem', False)

        ## load the data paths in this class
        self.lq_paths, self.gt_paths, self.ratios = [], [], []
        for root_folder, data_pair_list_ in zip(self.root_folders, self.data_pair_lists):
            with open(data_pair_list_, 'r') as data_pair_list:
                pairs = data_pair_list.readlines()
                for pair in pairs:
                    lq, gt = pair.split(' ')[:2]
                    gt = gt.rstrip('\n')
                    if ".ARW" in lq or ".nef" in lq or '.CR2' in lq: 
                        ratio = calculate_ratio(lq, gt, camera="Sony")
                    else:
                        ratio = calculate_ratio(lq, gt)
                    self.lq_paths.append(osp.join(root_folder, lq))
                    self.gt_paths.append(osp.join(root_folder, gt))
                    self.ratios.append(ratio)

        if self.load_in_mem:
            # load data in mem
            self.lqs = {
                data_path: depack_meta(data_path)
                for data_path in tqdm(set(self.lq_paths), desc='load lq metas in mem...')
            }
            self.gts = {
                data_path: depack_meta(data_path)
                for data_path in tqdm(set(self.gt_paths), desc='load gt metas in mem...')
            }

        self.len_lq_paths = len(self.lq_paths)



    def __getitem__(self, index):
        if self.noise_maker: index = index % self.len_lq_paths
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.ratios[index]

        if not self.load_in_mem:
            gt_im, gt_wb, gt_ccm, gt_scale, gt_white_level, gt_black_level = depack_meta(gt_path)
            if self.noise_maker:
                flag = random.randint(0, 1)
                if flag:
                    lq_im = self.noise_maker(gt_im)
                else:
                    lq_im, _, _, _, lq_white_level, lq_black_level= depack_meta(lq_path)
            else:
                lq_im, _, _, _, lq_white_level, lq_black_level = depack_meta(lq_path)
            
        else:
            gt_im, gt_wb, gt_ccm, gt_scale, gt_white_level, gt_black_level = self.gts[gt_path]
            if self.noise_maker:
                flag = random.randint(0, 1)
                if flag:
                    camera_id = torch.randint(0, len(self.noise_maker), (1,)).item()
                    _, lq_im, curr_metadata = self.noise_maker(gt_im, gt_scale, ratio, camera_id)
                else:
                    lq_im, _, _, _, lq_white_level, lq_black_level= self.lqs[lq_path]
            else:
                lq_im, _, _, _, lq_white_level, lq_black_level= self.lqs[lq_path]

        ### augment
        ## crop
        if self.opt['crop_size'] is not None:
            _, H, W = lq_im.shape
            crop_size = self.opt['crop_size']
            assert crop_size <= H and crop_size <= W
            if self.opt['phase'] == 'train':
                h_start = torch.randint(0, H - crop_size, (1,)).item()
                w_start = torch.randint(0, W - crop_size, (1,)).item()
            else:
                # center crop
                h_start = (H - crop_size) // 2
                w_start = (W - crop_size) // 2
            lq_im_patch = lq_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
            gt_im_patch = gt_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
        else:
            lq_im_patch = lq_im
            gt_im_patch = gt_im




        ## flip + rotate
        if self.opt['phase'] == 'train':
            hflip = self.opt['use_hflip'] and torch.rand((1,)).item() < 0.5
            vflip = self.opt['use_rot'] and torch.rand((1,)).item() < 0.5
            rot90 = self.opt['use_rot'] and torch.rand((1,)).item() < 0.5
            if hflip:
                lq_im_patch = torch.flip(lq_im_patch, (2,))
                gt_im_patch = torch.flip(gt_im_patch, (2,))
            if vflip:
                lq_im_patch = torch.flip(lq_im_patch, (1,))
                gt_im_patch = torch.flip(gt_im_patch, (1,))
            if rot90:
                lq_im_patch = torch.permute(lq_im_patch, (0, 2, 1))
                gt_im_patch = torch.permute(gt_im_patch, (0, 2, 1))

        lq_im_patch = torch.clip(lq_im_patch * ratio, None, 1)
        gt_im_patch = torch.clip(gt_im_patch, 0, 1)



        return {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': torch.tensor(ratio).float(),
            'wb': gt_wb,
            'ccm': gt_ccm,
            'scale': gt_scale,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return self.len_lq_paths * 2 if self.noise_maker else self.len_lq_paths
    


class MIPITestDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folders = opt['dataroot']
        self.load_in_mem = opt.get('load_in_mem', False)

        ## load the data paths in this class
        self.lq_paths = []
        for root_folder in self.root_folders:
            self.lq_paths.extend(sorted(glob(osp.join(root_folder, "*.npz"))))
        self.ratios = []
        for lq_path in self.lq_paths:
            lq = lq_path.split('/')[-1]
            ratio = calculate_ratio(lq)
            self.ratios.append(ratio)

        if self.load_in_mem:
            # load data in mem
            self.lqs = {
                data_path: depack_meta(data_path)
                for data_path in tqdm(set(self.lq_paths), desc='load lq metas in mem...')
            }

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        ratio = self.ratios[index]

        if not self.load_in_mem:
            lq_im, lq_wb, lq_ccm, lq_scale, white_level, black_level = depack_meta(lq_path)
        else:
            lq_im, lq_wb, lq_ccm, lq_scale, white_level, black_level = self.lqs[lq_path]

        lq_im = torch.clip(lq_im * ratio, None, 1)


        return {
            'lq': lq_im,
            'ratio': torch.tensor(ratio).float(),
            'wb': lq_wb,
            'ccm': lq_ccm,
            'lq_path': lq_path,
            'white_level': white_level,
            'black_level': black_level,
        }
    
    def __len__(self):
        return len(self.lq_paths)


class NoiseModelBase:  # base class
    def __call__(self, y, params=None):
        if params is None:
            K, g_scale, saturation_level, ratio = self._sample_params()
        else:
            K, g_scale, saturation_level, ratio = params

        y = y * saturation_level
        y = y / ratio
        
        if 'P' in self.model:
            z = np.random.poisson(y / K).astype(np.float32) * K
        elif 'p' in self.model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
        else:
            z = y

        if 'g' in self.model:
            z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10)  # Gaussian noise

        z = z * ratio
        z = z / saturation_level
        return z
    

# Only support baseline noise models: G / G+P / G+P* 
class NoiseModel(NoiseModelBase):
    def __init__(self, model='g', cameras=None, include=None, exclude=None, camera_params_path=""):
        super().__init__()
        assert include is None or exclude is None
        self.cameras = cameras or ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']        

        if include is not None:
            self.cameras = [self.cameras[include]]
        if exclude is not None:
            exclude_camera = set([self.cameras[exclude]])
            self.cameras = list(set(self.cameras) - exclude_camera)

        self.param_dir = join(camera_params_path, 'release')

        print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] cameras: {}'.format(self.cameras))
        print('[i] using noise model {}'.format(model))
        
        self.camera_params = {}

        for camera in self.cameras:
            self.camera_params[camera] = np.load(join(self.param_dir, camera+'_params.npy'), allow_pickle=True).item()

        self.model = model

    def _sample_params(self):
        camera = np.random.choice(self.cameras)
        # print(camera)

        saturation_level = (16383 - 800) * 100
        profiles = ['Profile-1']

        camera_params = self.camera_params[camera]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        # log_K = np.random.uniform(low=np.log(1e-1), high=np.log(30))
        
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
             camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']

        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)

        ratio = np.random.uniform(low=100, high=300)

        return (K, g_scale, saturation_level, ratio)



# # Only support baseline noise models: G / G+P / G+P* 
# class NoiseModel:
#     def __init__(self, model='g'):
#         super().__init__()
#         self.model = model
    
#     def __call__(self, y, params=None):
        
#         print("y.max(): ", y.max())
#         print("y.min(): ", y.min())
#         print("y.mean(): ", y.mean())

#         if 'P' in self.model:
#             z = np.random.poisson(y).astype(np.float32)
#         elif 'p' in self.model:
#             z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y, 1e-10))
#         else:
#             z = y

#         print("z.max(): ", z.max())
#         print("z.min(): ", z.min())
#         print("z.mean(): ", z.mean())

#         if 'g' in self.model:
#             z = z + np.random.randn(*y.shape).astype(np.float32)  # Gaussian noise

#         print("z.max(): ", z.max())
#         print("z.min(): ", z.min())
#         print("z.mean(): ", z.mean())

#         return z
    
def _pack_bayer(raw):
    h, w = raw.size(-2), raw.size(-1)
    out = torch.cat((raw[..., None, 0:h:2, 0:w:2], # R
                     raw[..., None, 0:h:2, 1:w:2], # G1
                     raw[..., None, 1:h:2, 1:w:2], # B
                     raw[..., None, 1:h:2, 0:w:2]  # G2
                    ), dim=-3)
    return out
    
def shot_noise(x, k):
    return torch.poisson(x / k) * k - x

def gaussian_noise(x, scale, loc=0):
    return torch.randn_like(x) * scale + loc

def tukey_lambda_noise(x, scale, t_lambda=1.4):
    def tukey_lambda_ppf(p, t_lambda):
        assert not torch.any(t_lambda == 0.0)
        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)

    epsilon = 1e-10
    U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = tukey_lambda_ppf(U, t_lambda) * scale

    return Y

def row_noise(x, scale, loc=0):
    if x.dim() == 4:
        B, _, H, W = x.shape
        noise = (torch.randn((B, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, W * 2))
        return _pack_bayer(noise)
    elif x.dim() == 5:
        B, T, _, H, W = x.shape
        noise = (torch.randn((B, T, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, 1, W * 2))
        return _pack_bayer(noise)
    else:
        raise NotImplementedError()


def quant_noise(x, q):
    return (torch.rand_like(x) - 0.5) * q

def _uniform_batch(min_, max_, shape=(1,), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.rand(shape, device=device) * (max_ - min_) + min_

def _normal_batch(scale=1.0, loc=0.0, shape=(1,), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.randn(shape, device=device) * scale + loc

def _randint_batch(min_, max_, shape=(1,), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.randint(min_, max_, shape, device=device)


class CalibratedNoisyPairGenerator(nn.Module):
    def __init__(
            self, 
            camera_params, 
            noise_type,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ) -> None:
        super().__init__()
        self.device = device
        self.camera_params = camera_params
        self.camera_params = self.param_dict_to_tensor_dict(self.camera_params)
        self.cameras = list(self.camera_params.keys())
        print('Current Using Cameras: ', self.cameras)

        self.noise_type = noise_type.lower()
        self.read_type = 'TukeyLambda' if 't' in self.noise_type else \
            ('Gaussian' if 'g' in self.noise_type else None)

    def param_dict_to_tensor_dict(self, p_dict):
        def to_tensor_dict(p_dict):
            for k, v in p_dict.items():
                if isinstance(v, list) or isinstance(v, Number):
                    p_dict[k] = nn.Parameter(torch.tensor(v, device=self.device), False)
                elif isinstance(v, dict):
                    p_dict[k] = to_tensor_dict(v)
            return p_dict
        return to_tensor_dict(p_dict)
    

    def sample_overall_system_gain(self, batch_size, for_video):
        if self.index is None:
            self.index = _randint_batch(0, len(self.camera_params)).item()
        self.current_camera = self.cameras[self.index]
        self.current_camera_params = self.camera_params[self.current_camera]
        self.current_k_range = [
            self.camera_params[self.current_camera]['Kmin'],
            self.camera_params[self.current_camera]['Kmax']
        ]
        log_K_max = torch.log(self.current_camera_params['Kmax'])
        log_K_min = torch.log(self.current_camera_params['Kmin'])
        log_K = _uniform_batch(log_K_min, log_K_max, (batch_size, 1, 1, 1), self.device)
        if for_video:
            log_K = log_K.unsqueeze(-1)
        self.log_K = log_K
        self.cur_batch_size = batch_size
        return torch.exp(log_K)

    def sample_read_sigma(self):
        slope = self.current_camera_params[self.read_type]['slope']
        bias = self.current_camera_params[self.read_type]['bias']
        sigma = self.current_camera_params[self.read_type]['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_tukey_lambda(self, batch_size, for_video):
        index = _randint_batch(0, len(self.current_camera_params[self.read_type]['lambda']), shape=(batch_size,))
        tukey_lambdas = self.current_camera_params[self.read_type]['lambda'][index].reshape(batch_size, 1, 1, 1)
        if for_video:
            tukey_lambdas = tukey_lambdas.unsqueeze(1)
        return tukey_lambdas

    def sample_row_sigma(self):
        slope = self.current_camera_params['Row']['slope']
        bias = self.current_camera_params['Row']['bias']
        sigma = self.current_camera_params['Row']['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.squeeze(-3).shape)

    def sample_color_bias(self, batch_size, for_video):
        count = len(self.current_camera_params['ColorBias'])
        i_range = (self.current_k_range[1] - self.current_k_range[0]) / count
        index = ((torch.exp(self.log_K.squeeze()) - self.current_k_range[0]) // i_range).long()
        color_bias = self.current_camera_params['ColorBias'][index]
        color_bias = color_bias.reshape(batch_size, 4, 1, 1)
        if for_video:
            color_bias = color_bias.unsqueeze(1)
        return color_bias

    @staticmethod
    def add_noise(img, noise, noise_params):
        tail = [1 for _ in range(img.dim() - 1)]
        ratio = noise_params['isp_dgain'].view(-1, *tail)
        scale = noise_params['scale'].view(-1, 4, *tail[:-1])
        for n in noise.values():
            img += n
        img /= scale
        img = img * ratio
        return torch.clamp(img, max=1.0)
    
    @torch.no_grad()
    def forward(self, img, scale, ratio, vcam_id=None):
        b = img.size(0)
        for_video = True if img.dim() == 5 else False # B, T, C, H, W
        self.index = vcam_id if vcam_id is not None else None

        img_gt = torch.clamp(img, 0, 1)
        tail = [1 for _ in range(img.dim() - 1)]
        img = img_gt * scale.view(-1, 4, *tail[:-1]) / ratio.view(-1, *tail)

        K = self.sample_overall_system_gain(b, for_video)
        noise = {}
        noise_params = {'isp_dgain': ratio, 'scale': scale}
        # shot noise
        if 'p' in self.noise_type:
            _shot_noise = shot_noise(img, K)
            noise['shot'] = _shot_noise
            noise_params['shot'] = K.squeeze()
        # read noise
        if 'g' in self.noise_type:
            read_param = self.sample_read_sigma()
            _read_noise = gaussian_noise(img, read_param)
            noise['read'] = _read_noise
            noise_params['read'] = read_param.squeeze()
        elif 't' in self.noise_type:
            tukey_lambda = self.sample_tukey_lambda(b, for_video)
            read_param = self.sample_read_sigma()
            _read_noise = tukey_lambda_noise(img, read_param, tukey_lambda)
            noise['read'] = _read_noise
            noise_params['read'] = {
                'sigma': read_param,
                'tukey_lambda': tukey_lambda
            }

        # row noise
        if 'r' in self.noise_type:
            row_param = self.sample_row_sigma()
            _row_noise = row_noise(img, row_param)
            noise['row'] = _row_noise
            noise_params['row'] = row_param.squeeze()
        # quant noise
        if 'q' in self.noise_type:
            _quant_noise = quant_noise(img, 1)
            noise['quant'] = _quant_noise
        # color bias
        if 'c' in self.noise_type:
            color_bias = self.sample_color_bias(b, for_video)
            noise['color_bias'] = color_bias

        img_lq = self.add_noise(img, noise, noise_params)

        return img_gt, img_lq, {
            'cam': self.current_camera,
            'noise': noise,
            'noise_params': noise_params
        }

    def __len__(self):
        return len(self.cameras)
    
    def cpu(self):
        super().cpu()
        self.device = 'cpu'
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self.device = 'cuda'
        return self

    @property
    def log_str(self):
        return f'{self._get_name()}: {self.cameras}'
    



class VirtualNoisyPairGenerator(nn.Module):
    def __init__(self,
                noise_type,
                param_ranges,
                sample_strategy,
                virtual_camera_count,
                shuffle = False,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        super().__init__()
        self.noise_type = noise_type.lower()
        self.param_ranges = param_ranges
        self.virtual_camera_count = virtual_camera_count
        self.sample_strategy = sample_strategy
        self.shuffle = shuffle
        self.device = device
        

        self.sample_virtual_cameras()

        print('Current Using Cameras: ', [f'IC{i}' for i in range(self.virtual_camera_count)])

    def sample_virtual_cameras(self):

        # sampling strategy
        sample = self.split_range if self.sample_strategy == 'coverage' else self.uniform_range

        # overall system gain
        self.k_range = torch.tensor(self.param_ranges['K'], device=self.device)

        # read noise
        if 'g' in self.noise_type:
            read_slope_range = self.param_ranges['Gaussian']['slope']
            read_bias_range = self.param_ranges['Gaussian']['bias']
            read_sigma_range = self.param_ranges['Gaussian']['sigma']
        elif 't' in self.noise_type:
            read_slope_range = self.param_ranges['TukeyLambda']['slope']
            read_bias_range = self.param_ranges['TukeyLambda']['bias']
            read_sigma_range = self.param_ranges['TukeyLambda']['sigma']
            read_lambda_range = self.param_ranges['TukeyLambda']['lambda']
            self.tukey_lambdas = sample(self.virtual_camera_count, read_lambda_range, self.shuffle, self.device)
            self.tukey_lambdas = nn.Parameter(self.tukey_lambdas, False)
        if 'g' in self.noise_type or 't' in self.noise_type:
            self.read_slopes = sample(self.virtual_camera_count, read_slope_range, self.shuffle, self.device)
            self.read_biases = sample(self.virtual_camera_count, read_bias_range, self.shuffle, self.device)
            self.read_sigmas = sample(self.virtual_camera_count, read_sigma_range, self.shuffle, self.device)
            self.read_slopes = nn.Parameter(self.read_slopes, False)
            self.read_biases = nn.Parameter(self.read_biases, False)
            self.read_sigmas = nn.Parameter(self.read_sigmas, False)

        # row noise
        if 'r' in self.noise_type:
            row_slope_range = self.param_ranges['Row']['slope']
            row_bias_range = self.param_ranges['Row']['bias']
            row_sigma_range = self.param_ranges['Row']['sigma']
            self.row_slopes = sample(self.virtual_camera_count, row_slope_range, self.shuffle, self.device)
            self.row_biases = sample(self.virtual_camera_count, row_bias_range, self.shuffle, self.device)
            self.row_sigmas = sample(self.virtual_camera_count, row_sigma_range, self.shuffle, self.device)
            self.row_slopes = nn.Parameter(self.row_slopes, False)
            self.row_biases = nn.Parameter(self.row_biases, False)
            self.row_sigmas = nn.Parameter(self.row_sigmas, False)

        # color bias
        if 'c' in self.noise_type:
            self.color_bias_count = self.param_ranges['ColorBias']['count']
            ## ascend sigma
            color_bias_sigmas = self.split_range_overlap(self.color_bias_count,
                                                         self.param_ranges['ColorBias']['sigma'],
                                                         overlap=0.1)
            self.color_biases = torch.tensor(np.array([
                [
                    random.uniform(*self.param_ranges['ColorBias']['bias']) + \
                        torch.randn(4).numpy() * random.uniform(*color_bias_sigmas[i]).cpu().numpy()
                    for _ in range(self.color_bias_count)
                ] for i in range(self.virtual_camera_count)
            ]), device=self.device)
            self.color_biases = nn.Parameter(self.color_biases, False)

    @staticmethod
    def uniform_range(splits, range_, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        results = [random.uniform(*range_) for _ in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results, device=device)

    @staticmethod
    def split_range(splits, range_, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        length = range_[1] - range_[0]
        i_length = length / (splits - 1)
        results = [range_[0] + i_length * i for i in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results, device=device)

    @staticmethod
    def split_range_overlap(splits, range_, overlap=0.5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        length = range_[1] - range_[0]
        i_length = length / (splits * (1 - overlap) + overlap)
        results = []
        for i in range(splits):
            start = i_length * (1 - overlap) * i
            results.append([start, start + i_length])
        return torch.tensor(results, device=device)

    def sample_overall_system_gain(self, batch_size, for_video):
        if self.current_camera is None:
            index = _randint_batch(0, self.virtual_camera_count, (batch_size,), self.device)
            self.current_camera = index
        log_K_max = torch.log(self.k_range[1])
        log_K_min = torch.log(self.k_range[0])
        log_K = _uniform_batch(log_K_min, log_K_max, (batch_size, 1, 1, 1), self.device)
        if for_video:
            log_K = log_K.unsqueeze(-1)
        self.log_K = log_K
        self.cur_batch_size = batch_size
        return torch.exp(log_K)

    def sample_read_sigma(self):
        slope = self.read_slopes[self.current_camera]
        bias = self.read_biases[self.current_camera]
        sigma = self.read_sigmas[self.current_camera]
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_tukey_lambda(self, batch_size, for_video):
        tukey_lambda = self.tukey_lambdas[self.current_camera].reshape(batch_size, 1, 1, 1)
        if for_video:
            tukey_lambda = tukey_lambda.unsqueeze(-1)
        return tukey_lambda

    def sample_row_sigma(self):
        slope = self.row_slopes[self.current_camera]
        bias = self.row_biases[self.current_camera]
        sigma = self.row_sigmas[self.current_camera]
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.squeeze(-3).shape)

    def sample_color_bias(self, batch_size, for_video):
        i_range = (self.k_range[1] - self.k_range[0]) / self.color_bias_count
        index = ((torch.exp(self.log_K.squeeze()) - self.k_range[0]) // i_range).long()
        color_bias = self.color_biases[self.current_camera, index]
        color_bias = color_bias.reshape(batch_size, 4, 1, 1)
        if for_video:
            color_bias = color_bias.unsqueeze(1)
        return color_bias

    @staticmethod
    def add_noise(img, noise, noise_params):
        tail = [1 for _ in range(img.dim() - 1)]
        ratio = noise_params['isp_dgain'].view(-1, *tail)
        scale = noise_params['scale'].view(-1, 4, *tail[:-1])
        for n in noise.values():
            img += n
        img /= scale
        img = img * ratio
        return torch.clamp(img, max=1.0)

    @torch.no_grad()
    def forward(self, img, scale, ratio, vcam_id=None):
        b = img.size(0)
        for_video = True if img.dim() == 5 else False # B, T, C, H, W
        self.current_camera = vcam_id * torch.ones((b,), dtype=torch.long, device=self.device) \
                              if vcam_id is not None else None

        img_gt = torch.clamp(img, 0, 1)
        tail = [1 for _ in range(img.dim() - 1)]
        img = img_gt * scale.view(-1, 4, *tail[:-1]) / ratio.view(-1, *tail)

        K = self.sample_overall_system_gain(b, for_video)
        noise = {}
        noise_params = {'isp_dgain': ratio, 'scale': scale}
        # shot noise
        if 'p' in self.noise_type:
            _shot_noise = shot_noise(img, K)
            noise['shot'] = _shot_noise
            noise_params['shot'] = K.squeeze()
        # read noise
        if 'g' in self.noise_type:
            read_param = self.sample_read_sigma()
            _read_noise = gaussian_noise(img, read_param)
            noise['read'] = _read_noise
            noise_params['read'] = read_param.squeeze()
        elif 't' in self.noise_type:
            tukey_lambda = self.sample_tukey_lambda(b, for_video)
            read_param = self.sample_read_sigma()
            _read_noise = tukey_lambda_noise(img, read_param, tukey_lambda)
            noise['read'] = _read_noise
            noise_params['read'] = {
                'sigma': read_param,
                'tukey_lambda': tukey_lambda
            }
        # row noise
        if 'r' in self.noise_type:
            row_param = self.sample_row_sigma()
            _row_noise = row_noise(img, row_param)
            noise['row'] = _row_noise
            noise_params['row'] = row_param.squeeze()
        # quant noise
        if 'q' in self.noise_type:
            _quant_noise = quant_noise(img, 1)
            noise['quant'] = _quant_noise
        # color bias
        if 'c' in self.noise_type:
            color_bias = self.sample_color_bias(b, for_video)
            noise['color_bias'] = color_bias

        img_lq = self.add_noise(img, noise, noise_params)

        return img_gt, img_lq, {
            'vcam_id': self.current_camera.squeeze(),
            'noise': noise,
            'noise_params': noise_params
        }

    def __len__(self):
        return self.virtual_camera_count

    def cpu(self):
        super().cpu()
        self.device = self.k_range.device
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self.device = self.k_range.device
        return self

    @property
    def json_dict(self):
        if hasattr(self, '_json_dict'):
            return self._json_dict

        json_dict = { f'IC{i}': {} for i in range(self.virtual_camera_count) }
        for i in range(self.virtual_camera_count):
            json_dict[f'IC{i}']['Kmin'] = self.k_range[0].cpu().numpy().tolist()
            json_dict[f'IC{i}']['Kmax'] = self.k_range[1].cpu().numpy().tolist()

        if 'g' in self.noise_type or 't' in self.noise_type:
            read_log_key = 'G' if 'g' in self.noise_type else 'TL'
            for i in range(len(self.read_slopes)):
                json_dict[f'IC{i}'][f'{read_log_key}_slope'] = self.read_slopes[i].cpu().numpy().tolist()
                json_dict[f'IC{i}'][f'{read_log_key}_bias'] = self.read_biases[i].cpu().numpy().tolist()
                json_dict[f'IC{i}'][f'{read_log_key}_sigma'] = self.read_sigmas[i].cpu().numpy().tolist()
                if read_log_key == 'TL':
                    json_dict[f'IC{i}'][f'{read_log_key}_lambda'] = self.tukey_lambdas[i].cpu().numpy().tolist()

        if 'r' in self.noise_type:
            for i in range(len(self.row_slopes)):
                json_dict[f'IC{i}']['Row_slope'] = self.row_slopes[i].cpu().numpy().tolist()
                json_dict[f'IC{i}']['Row_bias'] = self.row_biases[i].cpu().numpy().tolist()
                json_dict[f'IC{i}']['Row_sigma'] = self.row_sigmas[i].cpu().numpy().tolist()

        if 'c' in self.noise_type:
            for i in range(len(self.color_biases)):
                json_dict[f'IC{i}']['CB_biases'] = self.color_biases[i].cpu().numpy().tolist()

        self._json_dict = json_dict
        return json_dict

    @property
    def log_str(self):
        def clip_float_in_list(l, fmt=4, auto_newline=True):
            l_out = '['
            count = len(l)
            for i, f in enumerate(l):
                if torch.is_tensor(f):
                    f = f.cpu().numpy()
                if auto_newline and i % int(math.sqrt(count)) == 0 and not isinstance(f, np.ndarray):
                    l_out += '\n  '
                if isinstance(f, np.ndarray):
                    l_out += '\n  ' + str(np.array(f * 10 ** fmt, dtype='int') / float(10 ** fmt)) + ','
                else:
                    l_out += str(int(f * 10 ** fmt) / float(10 ** fmt)) + ', '
            else:
                if auto_newline:
                    l_out += '\n'
            l_out += ']'
            return l_out

        color_biases = deepcopy(self.color_biases)
        json_dict = deepcopy(self.json_dict)
        if 'c' in self.noise_type:
            for i in range(len(color_biases)):
                json_dict[f'IC{i}']['CB_biases'] = clip_float_in_list(color_biases[i])
        return tabulate(pd.DataFrame(json_dict).T, headers="keys", floatfmt='.4f')