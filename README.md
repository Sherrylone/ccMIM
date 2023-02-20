# Contextual Image Masking Modeling via Synergized Contrasting without View Augmentation for Faster and Better Visual Pretraining (ICLR 2023)

Code of ICLR 23 paper "[Contextual Image Masking Modeling via Synergized Contrasting without View Augmentation for Faster and Better Visual Pretraining](https://openreview.net/pdf?id=A3sgyt4HWp)"


![ccMIM](https://raw.githubusercontent.com/Sherrylone/sherrylone.github.io/main/figures/ccmim.png?token=GHSAT0AAAAAAB6QMNG5YWHAIG42QUH4VCEQY7TH7WA)

This paper presents to attentively masked semantic-richer patches by importance sampling strategy.

To pre-train the encoder on ImageNet-1K, run:

```
spring.submit arun -n 32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=5 --job-name ccMIM_pretrain\
 "python $ccMIM/submit_pretrain.py\
  --batch_size 32\
  --epochs 800\
  --model ccmim_vit_base_patch16\
  --mask_ratio 0.75\
  --world_size 32\
  --warmup_epochs 40 \
  --norm_pix_loss \
  --blr 1.5e-4 --weight_decay 0.05 \
  --output_dir $ccMIM/output/\
  --log_dir $ccMIM/output/\
  --mae false\
  --contrastive \
  --resume" &
```

The evaluation protocol follows MAE, which can be get in [MAE](https://github.com/facebookresearch/mae).

If you use ccMIM as baseline or find this repository useful, please consider citing our paper:

```
@inproceedings{
zhang2023contextual,
title={Contextual Image Masking Modeling via Synergized Contrasting without View Augmentation for Faster and Better Visual Pretraining},
author={Shaofeng Zhang and Feng Zhu and Rui Zhao and Junchi Yan},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=A3sgyt4HWp}
}
```
