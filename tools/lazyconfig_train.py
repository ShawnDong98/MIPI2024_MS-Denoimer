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
from torch.cuda.amp import autocast, GradScaler

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configs
args = default_argument_parser().parse_args()
cfg = LazyConfig.load(args.config_file)
cfg.DETERMINISTIC = True
logger, writer, output_dir = default_setup(cfg, args)



# Dataset
train_dataset = instantiate(cfg.dataset.train)
valid_dataset = instantiate(cfg.dataset.valid)
test_a_dataset = instantiate(cfg.dataset.test_a)



# Model
model = instantiate(cfg.model.arch).to(device)

# Optimizer
cfg.optimizer.params = model.parameters()
optimizer = instantiate(cfg.optimizer)

# Scheduler
cfg.scheduler.optimizer = optimizer
cfg.scheduler.num_warmup_steps=len(train_dataset) / cfg.dataset.train.opt.batch_size
cfg.scheduler.num_training_steps=(len(train_dataset) / cfg.dataset.train.opt.batch_size) * cfg.train.epochs
scheduler = instantiate(cfg.scheduler)

# EMA
cfg.model.ema.parameters = model.parameters()
ema = instantiate(cfg.model.ema)

if cfg.model.autocast: scaler = GradScaler()


if "gan_loss" in cfg.model.arch.losses: 
    cfg.model.D_optim.params = model.losses['gan_loss'].DNet.parameters()
    D_optim = instantiate(cfg.model.D_optim)


start_epoch = 0
if cfg.train.resume_ckpt_path:
    print(f"===> Loading Checkpoint from {cfg.train.resume_ckpt_path}")
    save_state = torch.load(cfg.train.resume_ckpt_path)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])
    optimizer.load_state_dict(save_state['optimizer'])
    scheduler.load_state_dict(save_state['scheduler'])
    start_epoch = save_state['epoch']


if cfg.test.pretrained_ckpt_path:
    print(f"===> Loading Checkpoint from {cfg.test.pretrained_ckpt_path}")
    save_state = torch.load(cfg.test.pretrained_ckpt_path)
    model.load_state_dict(save_state['model'], strict=False)
    if cfg.model.arch.pretrain:
        # EMA
        cfg.model.ema.parameters = model.parameters()
        ema = instantiate(cfg.model.ema)
    else:
        ema.load_state_dict(save_state['ema'])


def train(epoch, train_loader):
    model.train()
    epoch_loss = 0
    begin = time.time()
    train_tqdm = tqdm(enumerate(train_loader), total=len(train_dataset))

    for batch_idx, data in train_tqdm:
        data_time = time.time()
        data = {k:v.to(device) for k, v in data.items() if type(v) is torch.Tensor}
        data_time = time.time() - data_time

        model_time = time.time()
        if cfg.model.autocast:
            with torch.autocast(device_type=str(device)):
                loss_dict = model(data)
        else:
            loss_dict = model(data)
        model_time = time.time() - model_time

        loss_sum = 0
        for loss_name, loss in loss_dict.items(): 
            if loss_name == "gan_loss":
                loss_g = loss['loss_g']
                loss_d = loss['loss_d']
                loss_sum += loss_g
                D_optim.zero_grad()
            else:
                if loss_name == "tv_loss" and epoch >= 130:
                    continue
                loss_sum += loss
                loss_dict[loss_name] = f"{loss:.4f}"

        if epoch >= 130:
            loss_dict.pop("tv_loss")

        optimizer.zero_grad()
        if cfg.model.autocast:
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()

            if "gan_loss" in cfg.model.arch.losses:  
                scaler.scale(loss_d).backward()
                scaler.step(D_optim)
                scaler.update()
                loss_dict.pop("gan_loss")
                loss_dict['g_loss'] = f"{loss_g:.4f}"
                loss_dict['d_loss'] = f"{loss_d:.4f}"
        else:
            loss_sum.backward()
            optimizer.step()
            if "gan_loss" in cfg.model.arch.losses:  
                loss_d.backward()
                D_optim.step()
                loss_dict.pop("gan_loss")
                loss_dict['g_loss'] = f"{loss_g:.4f}"
                loss_dict['d_loss'] = f"{loss_d:.4f}"
        ema.update()
        loss_dict['data_time'] = data_time
        loss_dict['model_time'] = model_time
        train_tqdm.set_postfix(loss_dict)
        epoch_loss += loss_sum.data
        scheduler.step()
    end = time.time()
    train_loss = epoch_loss / len(train_dataset)
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, train_loss, (end - begin)))
    return train_loss


def eval(epoch, valid_loader):
    begin = time.time()
    model.eval()
    psnr_list, ssim_list, score_list = [], [], []
    train_tqdm = tqdm(enumerate(valid_loader), total=len(valid_dataset))
    for batch_idx, data in train_tqdm:
        data = {k:v.to(device) for k, v in data.items() if type(v) is torch.Tensor}
        if cfg.model.autocast:
            with torch.autocast(device_type=str(device)):
                with torch.no_grad():
                    with ema.average_parameters():
                        out = model(data)['pred']
        else:
            with torch.no_grad():
                    with ema.average_parameters():
                        out = model(data)['pred']

        score, psnr_v, ssim_v, rgb_out_torch, rgb_gt_torch = out2rgb_calculate_score(out, data['gt'], data['wb'], data['ccm'])
        psnr_list.append(psnr_v)
        ssim_list.append(ssim_v)
        score_list.append(score)

        rgb_out_torch = F.interpolate(rgb_out_torch, size=(128, 128))
        writer.add_image(f'images/valid/rgb_out_scene{batch_idx}', rgb_out_torch.squeeze(), epoch)

    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    score_mean = np.mean(np.asarray(score_list))

    end = time.time()

    writer.add_scalar('PSNR/valid', psnr_mean, epoch)
    writer.add_scalar('SSIM/valid', ssim_mean, epoch)
    writer.add_scalar('Score/valid', score_mean, epoch)

    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, score = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean, score_mean, (end - begin)))
    
    return psnr_mean, ssim_mean, score_mean


def test(epoch, test_loader):
    begin = time.time()
    model.eval()
    train_tqdm = tqdm(enumerate(test_loader), total=len(test_a_dataset))
    for batch_idx, data in train_tqdm:
        data = {k:v.to(device) for k, v in data.items() if type(v) is torch.Tensor}
        if cfg.model.autocast:
            with torch.autocast(device_type=str(device)):
                with torch.no_grad():
                    with ema.average_parameters():
                        out = model(data)['pred']
        else:
            with torch.no_grad():
                    with ema.average_parameters():
                        out = model(data)['pred']

        rgb_out_torch = out2rgb(out, data['wb'], data['ccm'])
        rgb_out_torch = F.interpolate(rgb_out_torch, size=(128, 128))
        writer.add_image(f'images/test_a/rgb_out_scene{batch_idx}', rgb_out_torch.squeeze(), epoch)
    
    end = time.time()

    logger.info('===> Epoch {}: testing time: {:.2f}'
                .format(epoch, (end - begin)))
    
    return out





def main():
    score_best = 0
    for epoch in range(start_epoch+1, cfg.train.epochs):
        train_loader = DataLoader(
                dataset = train_dataset,
                batch_size = cfg.dataset.train.opt.batch_size,
                shuffle = True,
                num_workers = cfg.dataset.train.opt.num_workers,
                pin_memory = False,
                drop_last = True
            )
        valid_loader = DataLoader(
                dataset = valid_dataset,
                batch_size = cfg.dataset.valid.opt.batch_size,
                shuffle = False,
                num_workers = cfg.dataset.valid.opt.num_workers,
                pin_memory = False,
                drop_last = False
            )
        train_loss = train(epoch, train_loader)
        torch.cuda.empty_cache()
        psnr, ssim, score = eval(epoch, valid_loader)

        test_loader = DataLoader(
            dataset = test_a_dataset,
            batch_size = cfg.dataset.test_a.opt.batch_size,
            shuffle = False,
            num_workers = cfg.dataset.test_a.opt.num_workers,
            pin_memory = False,
            drop_last = False
        )
        out = test(epoch, test_loader)

        if score > score_best:
            score_best = score
            checkpoint(model, ema, optimizer, scheduler, epoch, output_dir, logger)

if __name__ == "__main__":
    main()





