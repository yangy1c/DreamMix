<div align="center">

# DreamMix

### Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting

<p>
  <a href="https://mycfhs.github.io/"><strong>Yicheng Yang</strong></a>
  &nbsp;·&nbsp;
  <a href="https://pixeli99.github.io/">Pengxiang Li</a>
  &nbsp;·&nbsp;
  Lu Zhang
  &nbsp;·&nbsp;
  Liqian Ma
  &nbsp;·&nbsp;
  Ping Hu
  <br>
  Siyu Du
  &nbsp;·&nbsp;
  Yunzhi Zhuge
  &nbsp;·&nbsp;
  Xu Jia
  &nbsp;·&nbsp;
  Huchuan Lu
</p>

<p><em><strong>International Journal of Computer Vision (IJCV)</strong>, 2026</em></p>

<p>
  <a href="https://link.springer.com/article/10.1007/s11263-026-03066-w"><img alt="IJCV" src="https://img.shields.io/badge/IJCV-2026-1f6feb"></a>
  <a href="https://arxiv.org/abs/2411.17223"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2411.17223-b31b1b"></a>
  <a href="https://doi.org/10.1007/s11263-026-03066-w"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1007%2Fs11263--026--03066--w-0b7285"></a>
  <a href="https://scholar.google.com/citations?user=ghUWqRcAAAAJ"><img alt="Google Scholar" src="https://img.shields.io/badge/Google%20Scholar-Yicheng%20Yang-4285F4"></a>
  <a href="requirements.txt"><img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB"></a>
</p>

<img src="./assets/teaser.png" alt="Teaser" width="100%">

</div>

---

## News

- **[2026.10]** 🎉 DreamMix is accepted by **International Journal of Computer Vision (IJCV)**, Vol. 134, Issue 10, Article 464. [[Springer]](https://link.springer.com/article/10.1007/s11263-026-03066-w)
- **[2024.11]** Paper and code released on arXiv. [[arXiv]](https://arxiv.org/abs/2411.17223)

---

## Abstract

Subject-driven image inpainting has recently gained prominence in image editing with the rapid advancement of diffusion models. Beyond image guidance, recent studies have explored incorporating text guidance to achieve identity-preserved yet locally editable object inpainting. However, these methods still suffer from **identity overfitting**, where original attributes remain entangled with target textual instructions. To overcome this limitation, we propose **DreamMix**, a diffusion-based framework adept at inserting target objects into user-specified regions while concurrently enabling arbitrary text-driven attribute modifications. DreamMix introduces three key components:

- **(i) Attribute Decoupling Mechanism (ADM)** that synthesizes diverse attribute-augmented image-text pairs to mitigate overfitting;
- **(ii) Textual Attribute Substitution (TAS)** module that isolates target attributes via orthogonal decomposition;
- **(iii) Disentangled Inpainting Framework (DIF)** that separates local generation from global harmonization.

Extensive experiments across multiple inpainting backbones demonstrate that DreamMix achieves a superior balance between identity preservation and attribute editability across diverse applications, including object insertion, attribute editing, and small object inpainting.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Repository Structure

```text
DreamMix/
├── FooocusSDXLInpaintAllInOnePipeline.py  # Core inpainting pipeline
├── train.py                               # DreamBooth + DreamMix training script
├── infer.ipynb                            # Inference demo
├── make_img.ipynb                         # Regular-image synthesis
├── modules/                               # Sampling / utility modules adapted from Fooocus
├── ldm_patched/                           # Lightweight diffusion model layers
├── sdxl_styles/                           # Prompt style presets
├── models/                                # Pretrained weights (download manually)
├── data/                                  # Example inputs
└── assets/                                # Images used in this README
```

---

## Installation

> [!NOTE]
> DreamMix requires a GPU with **24 GB of memory** to run.

**1. Clone the repository**

```bash
git clone https://github.com/mycfhs/DreamMix.git
cd DreamMix
```

**2. Create the environment**

```bash
conda create -n DreamMix python=3.10
conda activate DreamMix
pip install -r requirements.txt
```

**3. Download the required models**

```bash
mkdir -p models/fooocus_inpaint models/upscale_models models/loras
```

| Model | Download | Destination |
| :-- | :-- | :-- |
| Fooocus inpaint v26 patch | [inpaint_v26.fooocus.patch](https://huggingface.co/lllyasviel/fooocus_inpaint/blob/main/inpaint_v26.fooocus.patch) | `models/fooocus_inpaint/` |
| Fooocus inpaint head | [fooocus_inpaint_head.pth](https://huggingface.co/lllyasviel/fooocus_inpaint/blob/main/fooocus_inpaint_head.pth) | `models/fooocus_inpaint/` |
| Upscale model | [fooocus_upscaler_s409985e5.bin](https://huggingface.co/metercai/SimpleSDXL/tree/main/upscale_models) | `models/upscale_models/` |
| Improve-quality LoRA *(optional)* | [sd_xl_offset_example-lora_1.0.safetensors](https://huggingface.co/metercai/SimpleSDXL/blob/main/loras/sd_xl_offset_example-lora_1.0.safetensors) | `models/loras/` |

---

## Data Preparation

**1. Install `lang-segment-anything`**

```bash
git clone https://github.com/mycfhs/lang-segment-anything
cd lang-segment-anything
pip install -e .
```

**2. Generate regular images**

Use the `make_img.ipynb` notebook to synthesize regular (background) images.

**3. Download the DreamBooth dataset**

Download the dataset [here](https://github.com/mycfhs/dreambooth_mask) and unpack it as `train_data/`:

```text
train_data/
└── <category>/
    └── image/
```

---

## Training

Launch DreamMix with `accelerate`:

```bash
CATEGORY="teapot"
INSTANCE_PROMPT="..."

accelerate launch train.py \
    --category="${CATEGORY}" \
    --output_dir="lora/${CATEGORY}" \
    --regular_dir="./regular_${CATEGORY}" \
    --regular_prob=0.3 \
    --loss_reweight_object=1.5 \
    --loss_reweight_background=0.6 \
    --pretrained_model_name_or_path="frankjoshua/juggernautXL_v8Rundiffusion" \
    --instance_data_dir="train_data/${CATEGORY}/image" \
    --mixed_precision="no" \
    --instance_prompt="${INSTANCE_PROMPT}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --seed="0" \
    --checkpointing_steps=250 \
    --resume_from_checkpoint="latest" \
    --enable_xformers_memory_efficient_attention
```

Key arguments:

| Argument | Description |
| :-- | :-- |
| `--category` | Object category to customize, also used as the output folder name. |
| `--instance_data_dir` | Directory containing the instance images. |
| `--regular_dir` | Directory containing the generated regular images. |
| `--loss_reweight_object` / `--loss_reweight_background` | Loss re-weighting for object and background regions. |
| `--regular_prob` | Probability of sampling a regular image in a training step. |
| `--checkpointing_steps` | Save a checkpoint every *N* steps. |

> [!TIP]
> Set `--resume_from_checkpoint="latest"` to resume from the most recent checkpoint; remove it to train from scratch.

---

## Inference

Follow the step-by-step instructions in [`infer.ipynb`](./infer.ipynb). The notebook loads the trained LoRA and runs the attribute-editable inpainting pipeline in `FooocusSDXLInpaintAllInOnePipeline.py`.

---

## Citation

If you find our work useful, please consider giving us a ⭐ and citing our paper:

```bibtex
@article{yang2026dreammix,
  title   = {DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting},
  author  = {Yang, Yicheng and Li, Pengxiang and Zhang, Lu and Ma, Liqian and Hu, Ping
             and Du, Siyu and Zhuge, Yunzhi and Jia, Xu and Lu, Huchuan},
  journal = {International Journal of Computer Vision},
  volume  = {134},
  number  = {10},
  pages   = {464},
  year    = {2026},
  doi     = {10.1007/s11263-026-03066-w}
}
```

<details>
<summary>arXiv version</summary>

```bibtex
@misc{yang2024dreammix,
  title         = {DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting},
  author        = {Yicheng Yang and Pengxiang Li and Lu Zhang and Liqian Ma and Ping Hu
                   and Siyu Du and Yunzhi Zhuge and Xu Jia and Huchuan Lu},
  year          = {2024},
  eprint        = {2411.17223},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2411.17223}
}
```

</details>

---

## Acknowledgements

We extend our gratitude to the open-source community. Our work is built upon the following resources:

- [Fooocus](https://github.com/lllyasviel/Fooocus) — for the fantastic inpaint method ([Inpaint v26 Fooocus Patch](https://huggingface.co/lllyasviel/fooocus_inpaint)).
- [JuggernautXL v8 Rundiffusion](https://huggingface.co/frankjoshua/juggernautXL_v8Rundiffusion) — our base generator.
- [Diffusers SDXL DreamBooth example](https://github.com/huggingface/diffusers/blob/v0.30.2/examples/dreambooth/train_dreambooth_lora_sdxl.py) — training code base.
- Sample images are collected from [Pixabay](https://pixabay.com/) and [COCO](https://cocodataset.org/).

Techniques incorporated from Fooocus:

- **Blur Guidance** — controlled with the `sharpness` parameter.
- **ADM Scaler** — parameters `adm_scaler_positive`, `adm_scaler_negative`, `adm_scaler_end`.
- **Inpaint Worker** — enhanced inpainting logic.
- **Prompt Style Enhancement** — improves prompt adaptability.
- **Advanced Sampler & Scheduler** — `Dpmpp2mSdeGpuKarras`.
- **Hybrid Models** — utilizes both base and inpainting models across different timesteps (`fooocus_time`).

---

## Contact

For questions or feedback, please reach out to **mycf2286247133@gmail.com**.
