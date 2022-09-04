---

<div align="center">    
 
# SePiCo: Semantic-Guided Pixel Contrast for Domain Adaptive Semantic Segmentation

[Binhui Xie](https://binhuixie.github.io), [Shuang Li](https://shuangli.xyz), [Mingjia Li](https://kiwixr.github.io), [Chi Harold Liu](https://scholar.google.com/citations?user=3IgFTEkAAAAJ&hl=en), [Gao Huang](http://www.gaohuang.net), and [Guoren Wang](https://scholar.google.com.hk/citations?hl=en&user=UjlGD7AAAAAJ)


[![Paper](http://img.shields.io/badge/paper-arxiv.2204.08808-B31B1B.svg)](https://arxiv.org/abs/2204.08808)

</div>

**Update on 2022/04/20: ArXiv Version of [SePiCo](https://arxiv.org/abs/2204.08808) is available.**

**Update on 2022/09/04: Code release.**

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sepico-semantic-guided-pixel-contrast-for/semantic-segmentation-on-dark-zurich&#41;]&#40;https://paperswithcode.com/sota/semantic-segmentation-on-dark-zurich?p=sepico-semantic-guided-pixel-contrast-for&#41;)

[//]: # ()
[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sepico-semantic-guided-pixel-contrast-for/unsupervised-domain-adaptation-on-synthia-to&#41;]&#40;https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-synthia-to?p=sepico-semantic-guided-pixel-contrast-for&#41;)

[//]: # ()
[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sepico-semantic-guided-pixel-contrast-for/synthetic-to-real-translation-on-gtav-to&#41;]&#40;https://paperswithcode.com/sota/synthetic-to-real-translation-on-gtav-to?p=sepico-semantic-guided-pixel-contrast-for&#41;)

[//]: # ()
[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sepico-semantic-guided-pixel-contrast-for/synthetic-to-real-translation-on-synthia-to-1&#41;]&#40;https://paperswithcode.com/sota/synthetic-to-real-translation-on-synthia-to-1?p=sepico-semantic-guided-pixel-contrast-for&#41;)



<!-- TOC -->

- [Overview](#overview)
- [Installation](#installation)
- [Datasets Preparation](#datasets-preparation)
  - [Download Datasets](#download-datasets)
  - [Setup Datasets](#setup-datasets)
- [Model Zoo](#model-zoo)
  - [GTAV &rarr; Cityscapes (DeepLab-v2 based)](#gtav--cityscapes-deeplab-v2-based)
  - [GTAV &rarr; Cityscapes (DAFormer based)](#gtav--cityscapes-daformer-based)
- [SePiCo Evaluation](#sepico-evaluation)
- [SePiCo Training](#sepico-training)
- [Tips on Code Understanding](#tips-on-code-understanding)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Contact](#contact)

<!-- /TOC -->

## Overview

In this work, we propose Semantic-Guided Pixel Contrast (SePiCo), a novel one-stage adaptation framework that highlights the semantic concepts of individual pixel to promote learning of class-discriminative and class-balanced pixel embedding space across domains, eventually boosting the performance of self-training methods.

<img src="resources/uda_results.png" width=50% height=50%>


<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## Installation

This code is implemented with `Python 3.8.5` and `PyTorch 1.7.1` on `CUDA 11.0`.

To try out this project, it is recommended to set up a virtual environment first:

```bash
# create and activate the environment
conda create --name sepico -y python=3.8.5
conda activate sepico

# install the right pip and dependencies for the fresh python
conda install -y ipython pip
```

Then, the dependencies can be installed by:

```bash
# install required packages
pip install -r requirements.txt

# install mmcv-full, this command compiles mmcv locally and may take some time
pip install mmcv-full==1.3.7  # requires other packeges to be installed first
```

**Alternatively**, the `mmcv-full` package can be installed faster with official pre-built packages, for instance:

```bash
# another way to install mmcv-full, faster
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

The environment is now fully prepared.

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## Datasets Preparation

### Download Datasets

- **GTAV:** Download all zipped images, along with their zipped labels, from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to a custom directory.
- **Cityscapes:** Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to a custom directory.

### Setup Datasets

Symlink the required datasets:

```bash
ln -s /path/to/gta5/dataset data/gta
ln -s /path/to/cityscapes/dataset data/cityscapes
```

Perform preprocessing to convert label IDs to the train IDs and gather dataset statistics:

```bash
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

Ultimately, the data structure should look like this:

```shell
SePiCo
├── ...
├── data
│   ├── cityscapes
│   │   ├── gtFine
│   │   ├── leftImg8bit
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## Model Zoo

We provide pretrained models of both Domain Adaptive Semantic Segmentation tasks through [Google Drive](https://drive.google.com/drive/folders/1dEkm3W79Wxoul6mqkpJP-ib1pKN0Mskm?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1jlEH_PvTtlQEHNN0k0Yf8A) (access code: `pico`).

### GTAV &rarr; Cityscapes (DeepLab-v2 based)

| variants | model name                       | mIoU  | checkpoint download                    |
| :------- | :------------------------------- | :---: | :------------------------------------- |
| DistCL   | sepico_distcl_gta2city_dlv2.pth  | 61.0  | [Google](https://drive.google.com/drive/folders/1_MBcTv7eUtoUFOe5Yfct7u7tvh2hUBMs?usp=sharing) / [Baidu](https://pan.baidu.com/s/13ezZyF9ZkClpWI83CfvdpA) (acc: `pico`) |
| BankCL   | sepico_bankcl_gta2city_dlv2.pth  | 59.8  | [Google](https://drive.google.com/drive/folders/1j_o_yT-bCD7JawSdlZI-BmfrOREEFgSq?usp=sharing) / [Baidu](https://pan.baidu.com/s/1_QoPttSu0GuDUzHhZ7pfYQ) (acc: `pico`) |
| ProtoCL  | sepico_protocl_gta2city_dlv2.pth | 58.8  | [Google](https://drive.google.com/drive/folders/1YiIsvNa3Fp1coBQwQ-DeAZd8VFolbjt-?usp=sharing) / [Baidu](https://pan.baidu.com/s/1nc81QwJooA9_mxFxteVCJg) (acc: `pico`) |

### GTAV &rarr; Cityscapes (DAFormer based)

| variants | model name                           | mIoU  | checkpoint download                    |
| :------- | :----------------------------------- | :---: | :------------------------------------- |
| DistCL   | sepico_distcl_gta2city_daformer.pth  | 70.3  | [Google](https://drive.google.com/drive/folders/1svqqz2vdELZSbFh4x96OCx0i1tRQRqWn?usp=sharing) / [Baidu](https://pan.baidu.com/s/1OFOxN1kG0KqKej_5_48iVQ) (acc: `pico`) |
| BankCL   | sepico_bankcl_gta2city_daformer.pth  | 68.7  | [Google](https://drive.google.com/drive/folders/15FAcxpRilaYe4woShSYWfpQli8TZ57n6?usp=sharing) / [Baidu](https://pan.baidu.com/s/1JScQYfgSYL16wYGLesLYug) (acc: `pico`) |
| ProtoCL  | sepico_protocl_gta2city_daformer.pth | 68.5  | [Google](https://drive.google.com/drive/folders/1035n8ZtK94gMg1M9eGOnl0K9to2WRg3I?usp=sharing) / [Baidu](https://pan.baidu.com/s/1rFxRLHa-kqNDWXao0VPrfA) (acc: `pico`) |


[//]: # (### SYNTHIA &rarr; Cityscapes &#40;DeepLab-v2 based&#41;)

[//]: # ()
[//]: # (### SYNTHIA &rarr; Cityscapes &#40;DAFormer based&#41;)

[//]: # ()
[//]: # (### Cityscapes &rarr; Dark Zurich &#40;DAFormer based&#41;)

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## SePiCo Evaluation

To evaluate the pretrained models, please run as follows:

```bash
python -m tools.test /path/to/config /path/to/checkpoint --eval mIoU
```

<details>
<summary>Example</summary>

For example, if you download `sepico_distcl_gta2city_dlv2.pth` along with its config json file `sepico_distcl_gta2city_dlv2.json` into folder `./checkpoints/sepico_distcl_gta2city_dlv2/`, then the evaluation script should be like:

```bash
python -m tools.test ./checkpoints/sepico_distcl_gta2city_dlv2/sepico_distcl_gta2city_dlv2.json ./checkpoints/sepico_distcl_gta2city_dlv2/sepico_distcl_gta2city_dlv2.pth --eval mIoU
```

</details>

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## SePiCo Training

To begin with, download [SegFormer](https://github.com/NVlabs/SegFormer)'s official MiT-B5 weights (i.e., `mit_b5.pth`) pretrained on ImageNet-1k from [here](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) and put it into a new folder `./pretrained`.

The training entrance is at `run_experiments.py`. To examine the setting for a specific task, please take a look at `experiments.py` for more details. Generally, the training script is given as:

```bash
python run_experiments.py --exp <exp_id>
```

All tasks are run on *GTAV &rarr; Cityscapes*, and the mapping between `<exp_id>` and tasks is:

| `<exp_id>` | variant | backbone   | feature    |
| :--------: | :------ | :--------- | :--------- |
|    `1`     | DistCL  | ResNet-101 | layer-4    |
|    `2`     | BankCL  | ResNet-101 | layer-4    |
|    `3`     | ProtoCL | ResNet-101 | layer-4    |
|    `4`     | DistCL  | MiT-B5     | all-fusion |
|    `5`     | BankCL  | MiT-B5     | all-fusion |
|    `6`     | ProtoCL | MiT-B5     | all-fusion |

After training, the models can be tested following [SePiCo Evaluation](#sepico-evaluation). Note that the training results are located in `./work_dirs`. The config filename should look like: `220827_1906_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_4cc9a.json`, and the model file has suffix `.pth`.

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## Tips on Code Understanding

- Class-balanced cropping (CBC) strategy is implemented as `RandomCrop` class in [mmseg/models/utils/ours_transforms.py](mmseg/models/utils/ours_transforms.py).
- The projection head can be found in [mmseg/models/decode_heads/proj_head.py](mmseg/models/decode_heads/proj_head.py).
- The semantic prototypes used for feature storage are implemented in [mmseg/models/utils/proto_estimator.py](mmseg/models/utils/proto_estimator.py), where all three variants of prototypes are included. For detailed usage, please refer to `mmseg/models/uda/sepico.py`.
- The losses in correspondence to the three variants of our framework, along with the regularization term, are implemented in [mmseg/models/losses/contrastive_loss.py](mmseg/models/losses/contrastive_loss.py).

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## Acknowledgments

This project is based on the following open-source projects. We thank their authors for making the source code publicly available.


- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (Apache License 2.0, [license details](resources/license_mmseg))
- [SegFormer](https://github.com/NVlabs/SegFormer) (NVIDIA Source Code License, [license details](resources/license_segformer))
- [DAFormer](https://github.com/lhoyer/DAFormer) (Apache License 2.0, [license details](resources/license_daformer))
- [DACS](https://github.com/vikolss/DACS) (MIT License, [license details](resources/license_dacs))

<div align="right">
<b><a href="#overview">↥</a></b>
</div>

## Citation

If you find this work helpful to your research, please consider citing the paper:

```bibtex
@article{xie2022sepico,
  title={SePiCo: Semantic-Guided Pixel Contrast for Domain Adaptive Semantic Segmentation},
  author={Xie, Binhui and Li, Shuang and Li, Mingjia and Liu, Chi Harold and Huang, Gao and Wang, Guoren},
  journal={arXiv preprint arXiv:2204.08808},
  year={2022}
}
```

<div align="right">
<b><a href="#overview">↥</a></b>
</div>


## Contact

If you have any questions about our code, feel free to contact us or describe your problem in [Issues](https://github.com/BIT-DA/SePiCo/issues/new).

<div align="right">
<b><a href="#overview">↥</a></b>
</div>
