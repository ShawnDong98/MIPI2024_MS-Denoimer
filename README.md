# MS-Denoimer: Multi Stage Denoiser with Spatial and Channel-wise Attention for Raw Image Denoising


# Abstract

# Architecture


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

# Train

# Test

Put the pretrained checkpoints into corresponding directories

Run following command

```
bash ensemble.sh
```

The results will be save at `results/test_save_img/`
 

# Ackownlege


