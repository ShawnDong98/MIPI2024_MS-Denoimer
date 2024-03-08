# MS-Denoimer: Multi Stage Denoiser with Spatial and Channel-wise Attention for Raw Image Denoising

This repo is the implementation of the solution "MS-Denoiser: Multi-Stage Denoiser with Spatial and Channel-wise Attention for Raw Image Denoising" in the CVPR 2024 workshop challenge MIPI2024 Few-shot Raw Image Denoising.

<div align=center>
<img src="https://github.com/ShawnDong98/MIPI2024_MS-Denoimer/blob/main/figures/teaser.gif" width = "350" height = "350" alt="">
</div>

# Abstract

In this challenge, we propose a Multi-Stage Denoiser with Spaital and Channe-wise Attention (MS-Denoimer) for raw image denoising. Specifically, we utilize a local and non-local multi-head self-attention mechanism to capture spatial correlations and a channel-wise multi-head self-attention mechanism to address channel-specific dependencies. These elements are compose basic units: Spatial Multi-head Self-Attention Blocks (S-MSABs) and Channel-wise Multi-head Self-Attention Blocks (C-MSABs), which are alternately build up the single-stage Denoising Transformer (Denoimer). The Denoimer exploits a U-shaped structure to extract multi-resolution contextual information.  Finally, our MS-Denoimer, cascaded by several Denoimers, progressively improves the reconstruction quality from coarse to fine. 

# Architecture

<div align=center>
<img src="https://github.com/ShawnDong98/MIPI2024_MS-Denoimer/blob/main/figures/MS-Denoimer.png" width = "700" height = "360" alt="">
</div>

Diagram of MS-Denoimer. (a) The diagram of the Multi-Stage Denoimer. (b) The diagram of the single-stage Denoimer. (c) The diagram of the Spatial Multi-head Self-attention Block (S-MSAB). (d) The diagram of the Channel-wise Multi-head Self-attention Block (C-MSAB). (e) The illustration of the Spatial Multi-head Self-Attention (S-MSA). (f) The illustration of the Channel-wise Multi-head Self-Attention (C-MSA). (g) The illustration of the Gated-DConv Feedforward Network (GDFN).


# Dataset

The dataset directory shows as following:

```
|---datasets
|---|---RawDenoising
|---|---|---train
|---|---|---|---Camera1
|---|---|---|---|---short
|---|---|---|---|---long
|---|---|---|---|---train.txt
|---|---|---|---Camera2
|---|---|---|---|---short
|---|---|---|---|---long
|---|---|---|---|---train.txt
|---|---|---valid
|---|---|---|---Camera1
|---|---|---|---|---short
|---|---|---|---|---valid.txt
|---|---|---|---Camera2
|---|---|---|---|---short
|---|---|---|---|---valid.txt
|---|---|---test
|---|---|---|---Camera1
|---|---|---|---|---short
|---|---|---|---Camera2
|---|---|---|---|---short
|---|---|---SID
|---|---|---|---Sony
|---|---|---|---|---short
|---|---|---|---|---long
|---|---|---|---|---train.txt
```


# Test

Put the pretrained checkpoints ([google drive]() | [baidu disk]()) into corresponding directories

Run following command

```
bash ensemble.sh
```

The results will be save at `results/test_save_img/`
 

# Acknowledgements

Our code is based on the following repo, thanks for their generous open source:

- [https://github.com/Srameo/LED](https://github.com/Srameo/LED)
- [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

