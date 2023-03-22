# Efficient Diffusion Training via Min-SNR Weighting Strategy

By Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, Baining Guo. 

[Paper](https://arxiv.org/abs/2303.09556) | [Code](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training)



**Abstract**. Denoising diffusion models have been a mainstream approach for image generation, however, training these models often suffers from slow convergence. In this paper, we discovered that the slow convergence is partly due to conflicting optimization directions between timesteps. To address this issue, we treat the diffusion training as a multi-task learning problem, and introduce a simple yet effective approach referred to as Min-SNR-$\gamma$. This method adapts loss weights of timesteps based on clamped signal-to-noise ratios, which effectively balances the conflicts among timesteps. Our results demonstrate a significant improvement in converging speed, 3.4x faster than previous weighting strategies. It is also more effective, achieving a new record FID score of 2.06 on the ImageNet 256x256 benchmark using smaller architectures than that employed in previous state-of-the-art.

## Data Preparation

For CelebA dataset, we follow [ScoreSDE](https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/datasets.py#L121) to process the data.

For ImageNet dataset, we download it from the [official website](https://www.image-net.org/). For ImageNet-64, we did not adopt offline pre-processing. For ImageNet-256, we cropped the images to 256x256 and compressed them using AutoencoderKL from [Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py).
The compressed latent codes are treated equally as images, except the file extension.

## Training
For training with ViT-B model, you should first put the downloaded/processed data above to some path, and set `DATA_DIR` in the config file [`vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs8x32.sh`](./configs/in256/vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs8x32.sh). Then you could run like
```bash
GPUS=8
BATCH_SIZE_PER_GPU=32
bash configs/in256/vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs8x32.sh $GPUS $BATCH_SIZE_PER_GPU
```


## Sampling with Pre-trained Models
For sampling for ImageNet-256, you could directly run
```bash
bash configs/in256/inference.sh
```

For sampling for ImageNet-64, you could directly run
```bash
bash configs/in64/inference.sh
```

Here we use 8 GPUs for sampling. You can change `GPUS=8` to `GPUS=1` for single GPU evaluation in [`configs/in256/inference.sh`](./configs/in256/inference.sh) 
The pre-trained models will be automatically downloaded and FID-50K will be calculated.

## Citing Min-SNR Diffusion Training
If you find our work useful for your research, please consider citing our paper. :blush:
```
@article{hang2023efficient,
      title={Efficient Diffusion Training via Min-SNR Weighting Strategy}, 
      author={Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
      year={2023},
      eprint={2303.09556},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowlegements
This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
We adopt the implementation for sampling and FID evaluation using [NVlabs/edm](https://github.com/NVlabs/edm).
