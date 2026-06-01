# KD-Mixup
Knowledge Distillation and mixup are widely used techniques for improving the efficiency and generalisation of deep neural networks. While KD transfers knowledge through soft targets produced by a teacher, mixup regularises training by interpolating inputs and labels.  Despite their success, their interaction remains poorly understood.

Based on this mixup-KD pipeline, a student model is trained to mimic a pre-trained teacher using soft targets combined with Mixup data augmentation.

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
