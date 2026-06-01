# KD-Mixup
Knowledge Distillation and mixup are widely used techniques for improving the efficiency and generalisation of deep neural networks. While KD transfers knowledge through soft targets produced by a teacher, mixup regularises training by interpolating inputs and labels.  Despite their success, their interaction remains poorly understood.

Based on this mixup-KD pipeline, a student model is trained to mimic a pre-trained teacher using soft targets combined with Mixup data augmentation.

---
# KD-Teachers

Pre-trained teacher models used for Knowledge Distillation with Mixup augmentation on CIFAR-10 and CIFAR-100. These checkpoints are the teacher component of the [KD-Mixup](https://github.com/JoseLMedinaC/KD-Mixup) framework.

All models were fine-tuned from ImageNet pre-trained weights on CIFAR-10 and CIFAR-100 using SGD with momentum, ReduceLROnPlateau scheduling, and mixed precision training (float16).

---

## Models

| Model | Backbone | Pre-training |
|---|---|---|
| `best_resnet152v2` | ResNet-152 V2 | ImageNet |
| `best_convnexttiny` | ConvNeXt-Tiny | ImageNet |
| `best_convnextlarge` | ConvNeXt-Large | ImageNet |
| `best_vitbase` | ViT-B/16 | ImageNet |

---

## Usage

Download a checkpoint and place it in your local `checkpoints/teachers/{dataset}/` folder:

```python
from huggingface_hub import hf_hub_download
import os
REPO_ID = "josemedina/KD-Teachers"
DATASET = "cifar100"  # "cifar10" o "cifar100"
TEACHER = "resnet152v2"  # "resnet152v2", "convnexttiny", "convnextlarge", "vitbase"
filename = f"{DATASET}/best_{TEACHER}.keras"
local_dir = f"checkpoints/teachers/{DATASET}"
os.makedirs(local_dir, exist_ok=True)
print(f"Downloading {filename}...")
path = hf_hub_download(
    repo_id=REPO_ID,
    filename=filename,
    local_dir="checkpoints/teachers"
)
print(f"Saved to: {path}")
```
---
## Training a Student Model

```bash
python -m scripts.student_train_mixup --teacher resnet152v2 --temperature 2 --student resnet50
```

### Arguments

| Argument | Required | Options | Default | Description |
|---|---|---|---|---|
| `--student` | ✅ | `resnet50`, `mobilenetv2` | — | Student model architecture |
| `--teacher` | ✅ | `resnet152v2`, `convnexttiny`, `convnextlarge`, `vitbase` | — | Teacher model architecture |
| `--temperature` | ❌ | any float | `2.0` | Distillation temperature. Use `0` for T=std (normalization by standard deviation) |

---

## Teacher Checkpoints

The teacher model must be available locally before training. Place the checkpoint at:

```
checkpoints/teachers/{dataset}/best_{teacher_name}.keras
```

For example:

```
checkpoints/teachers/cifar100/best_resnet152v2.keras
checkpoints/teachers/cifar10/best_convnexttiny.keras
```

Supported datasets: `cifar10`, `cifar100`

---

## Available Combinations

| Teacher | Student | Dataset |
|---|---|---|
| `resnet152v2` | `resnet50` / `mobilenetv2` | `cifar10` / `cifar100` |
| `convnexttiny` | `resnet50` / `mobilenetv2` | `cifar10` / `cifar100` |
| `convnextlarge` | `resnet50` / `mobilenetv2` | `cifar10` / `cifar100` |
| `vitbase` | `resnet50` / `mobilenetv2` | `cifar10` / `cifar100` |

---

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{medina2025kdmixup,
  author    = {Medina, Jos{\'e} and Honeine, Paul and Bensrhair, Abdelaziz and Hadachi, Amnir},
  title     = {Beyond Dark Knowledge: Mixup-Based Knowledge Distillation Under Vicinal Teacher Distributions},
  year      = {2025},
  publisher = {University of Tartu},
  url       = {https://github.com/JoseLMedinaC/KD-Mixup}
}
```