# DiffusionLight-Turbo: Accelerated Light Probes for Free via Single-Pass Chrome Ball Inpainting 	

### [Project Page](https://diffusionlight.github.io/turbo/) | [Paper](https://arxiv.org/abs/2312.09168) | [Colab](https://colab.research.google.com/drive/1UcSp9mj77ZXAyTCvkcXVvC3DYznEyLwZ?usp=sharing&sandboxMode=true#scrollTo=k2pTDk79bMQI&forceEdit=true&sandboxMode=true) | [HuggingFace](https://huggingface.co/DiffusionLight/DiffusionLight) | [ComfyUI](https://github.com/DiffusionLight/DiffusionLight-ComfyUI) | [Diffusers](https://github.com/DiffusionLight/Diffusionlight-turbo-diffusers)

[![Open DiffusionLight in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UcSp9mj77ZXAyTCvkcXVvC3DYznEyLwZ?usp=sharing&sandboxMode=true#scrollTo=k2pTDk79bMQI&forceEdit=true&sandboxMode=true)

![x60 faster! Same quality!](https://github.com/user-attachments/assets/ff658534-88cb-49dd-832c-2aebc159e9bd)


We introduce a simple yet effective technique for estimating lighting from a single low-dynamic-range (LDR) image by reframing the task as a chrome ball inpainting problem. This approach leverages a pre-trained diffusion model, Stable Diffusion XL, to overcome the generalization failures of existing methods that rely on limited HDR panorama datasets. While conceptually simple, the task remains challenging because diffusion models often insert incorrect or inconsistent content and cannot readily generate chrome balls in HDR format. Our analysis reveals that the inpainting process is highly sensitive to the initial noise in the diffusion process, occasionally resulting in unrealistic outputs. To address this, we first introduce DiffusionLight, which uses iterative inpainting to compute a median chrome ball from multiple outputs to serve as a stable, low-frequency lighting prior that guides the generation of a high-quality final result. To generate high-dynamic-range (HDR) light probes, an Exposure LoRA is fine-tuned to create LDR images at multiple exposure values, which are then merged. While effective, DiffusionLight is time-intensive, requiring approximately 30 minutes per estimation. To reduce this overhead, we introduce DiffusionLight-Turbo, which reduces the runtime to about 30 seconds with minimal quality loss. This 60x speedup is achieved by training a Turbo LoRA to directly predict the averaged chrome balls from the iterative process. Inference is further streamlined into a single denoising pass using a LoRA swapping technique. Experimental results that show our method produces convincing light estimates across diverse settings and demonstrates superior generalization to in-the-wild scenarios.

## Choosing Right Repository!

For Non-Researcher, we implemented other 2 different versions which are [ComfyUI](https://github.com/DiffusionLight/DiffusionLight-ComfyUI) and [Vanila diffusers](https://github.com/DiffusionLight/Diffusionlight-turbo-diffusers).

For Researcher, we use this repository to doing experiment, we include the code of turbo_sdedit, turbo_pred, and turbo_swapping for you to tinker around. 

## Table of contents
-----
  * [TL;DR](#Getting-started)
  * [Installation](#Installation)
  * [Prediction](#Prediction)
  * [Evaluation](#Evaluation)
  * [Citation](#Citation)
------

## Getting started

```shell
conda env create -f environment.yml
conda activate diffusionlight-turbo
pip install -r requirements.txt
python inpaint.py --dataset example --output_dir output
python ball2envmap.py --ball_dir output/square --envmap_dir output/envmap
python exposure2hdr.py --input_dir output/envmap --output_dir output/hdr
```

## Installation

To setup the Python environment, you need to run the following commands in both Conda and pip:

```shell
conda env create -f environment.yml
conda activate diffusionlight-turbo
pip install -r requirements.txt
```

Note that Conda is optional. However, if you choose not to use Conda, you must manually install CUDA-toolkit and OpenEXR. Or you can use singularity container definition files `singularity.def` as we provide.

## Prediction

### 0. Preparing the image

Please resize the input image to 1024x1024. If the image is not square, we recommend padding it with a black border.

### 1. Inpainting the chrome ball

First, we predict the chrome ball in different exposure values (EV) using the following command:

```shell
python inpaint.py --dataset <input_directory> --output_dir <output_directory>
```

This command outputs three subdirectories:  `raw`, and  `square`

The contents of each directory are:

- `raw`: Inpainted image with a chrome ball in the center
- `square`: Square-cropped chrome ball (used for the next step)


### 2. Projecting a ball into an environment map 

Next, we project the chrome ball from the previous step to the LDR environment map using the following command:

```shell
python ball2envmap.py --ball_dir <output_directory>/square --envmap_dir <output_directory>/envmap
```

### 3. Compose HDR image

Finally, we compose an HDR image from multiple LDR environment maps using our custom exposure bracketing:

```shell
python exposure2hdr.py --input_dir <output_directory>/envmap --output_dir <output_directory>/hdr
```

The predicted light estimation will be located at `<output_directory>/hdr` and can be used for downstream tasks such as object insertion. We will also use it to compare with other methods.

## Evaluation 
We use same evaluation method as the confernece version of DiffusionLight. Please visit evaluation repo  [DiffusionLight-evaluation](https://github.com/DiffusionLight/DiffusionLight-evaluation) for more information

## Citation

```
@inproceedings{Chinchuthakun2025DiffusionLightTurbo,
  author = {Chinchuthakun, Worameth and Phongthawee, Pakkapon and Raj, Amit and Jampani, Varun and Khungurn, Pramook and Suwajanakorn, Supasorn},
  title = {DiffusionLight-Turbo: Accelerated Light Probes for Free via Single-Pass Chrome Ball Inpainting},
  booktitle = {ArXiv},
  year = {2025},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)
