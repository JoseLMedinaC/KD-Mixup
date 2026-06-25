#!/usr/bin/env python3
"""
Calibration Evaluation teachers/STUDENTS on CIFAR.

- Accuracy
- Average confidence
- ECE
- Reliability diagram

Execution:
python -m scripts.ece --model resnet152v2 --classes 100 --bins 15 --batch 32
"""

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# ---------------------- GPU config -------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ---------------------- Argumentos CLI -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=[
    "resnet50", "resnet152v2", "convnexttiny", "convnextlarge", "mobilenetv2", "vitbase"
], help="Model to evaluate")
parser.add_argument("--teacher", default="teacher", type=str, choices=[
    "resnet152v2", "convnexttiny", "convnextlarge", "vitbase"
], help="Model to evaluate")
parser.add_argument("--classes", type=int, default=100, choices=[10, 100],
                    help="Dataset CIFAR10 or CIFAR100")
parser.add_argument("--img_size", type=int, default=224,
                    help="image size")
parser.add_argument("--batch", type=int, default=32,
                    help="Batch size")
parser.add_argument("--bins", type=int, default=15,
                    help="ECE bins number and reliability diagram")
parser.add_argument("--type", type=str, default="teachers",
                    help="teachers or students")
args = parser.parse_args()

MODEL_NAME = args.model.lower()
CLASSES = args.classes
IMG_SIZE = args.img_size
BATCH = args.batch
TYPES = args.type
N_BINS = args.bins
TEACHER = args.teacher

if TYPES=="teachers":
    model_ckpt_path = Path(f"checkpoints/{TYPES}/cifar{CLASSES}/best_{MODEL_NAME}.keras")
else:
    model_ckpt_path = Path(f"checkpoints/{TYPES}/cifar{CLASSES}/single_teacher/{MODEL_NAME}/T2/distill_{TEACHER}.keras")
txt_out_path = model_ckpt_path.parent / f"ece_{MODEL_NAME}_{TEACHER}.txt"
plot_out_path = model_ckpt_path.parent / f"reliability_{MODEL_NAME}_{TEACHER}.png"


# --------------------------- Dataset ---------------------------
def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if MODEL_NAME == "vitbase" and TYPES=="teachers":
        # Normalización ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        img = tf.cast(img, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        img  = (img - mean) / std
    return img, tf.one_hot(label, CLASSES)

_, val_ds = tfds.load(
    f"cifar{CLASSES}",
    split=["train", "test"],
    as_supervised=True
)

val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)


# --------------------------- Predicciones ---------------------------
def collect_predictions(model, dataset):
    y_true_all = []
    y_prob_all = []

    for x_batch, y_batch in dataset:
        outputs = model(x_batch, training=False)
        outputs = tf.cast(outputs, tf.float32).numpy()

        # Si el modelo ya entrega probabilidades, usarlas directamente.
        #probs = outputs
        probs = tf.nn.softmax(outputs, axis=1).numpy()
        # Solo si estuvieras seguro de que el modelo devuelve logits:
        # probs = tf.nn.softmax(tf.convert_to_tensor(outputs), axis=1).numpy()

        y_true = tf.argmax(y_batch, axis=1).numpy()

        y_true_all.append(y_true)
        y_prob_all.append(probs)

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    return y_true_all, y_prob_all


# --------------------------- Métricas ---------------------------
def compute_calibration_stats(y_true, y_prob, n_bins=15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correctness = (predictions == y_true).astype(np.float32)

    accuracy = float(np.mean(correctness))
    avg_confidence = float(np.mean(confidences))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    bin_centers = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = int(np.sum(in_bin))
        prop = np.mean(in_bin)

        center = (lower + upper) / 2.0
        bin_centers.append(center)
        bin_counts.append(count)

        if count > 0:
            acc_in_bin = float(np.mean(correctness[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(acc_in_bin - conf_in_bin) * prop
        else:
            acc_in_bin = 0.0
            conf_in_bin = 0.0

        bin_accs.append(acc_in_bin)
        bin_confs.append(conf_in_bin)

    return {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "ece": float(ece),
        "bin_centers": np.array(bin_centers),
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
    }


# --------------------------- Reliability Diagram ---------------------------
def plot_reliability_diagram(stats, save_path, model_name, classes, n_bins):
    bin_centers = stats["bin_centers"]
    bin_accs = stats["bin_accs"]
    bin_confs = stats["bin_confs"]
    accuracy = stats["accuracy"]
    avg_confidence = stats["avg_confidence"]
    ece = stats["ece"]

    width = 1.0 / n_bins

    plt.figure(figsize=(7, 7))

    # Barras de accuracy por bin
    plt.bar(
        bin_centers,
        bin_accs,
        width=width,
        alpha=0.7,
        edgecolor="black",
        label="Accuracy"
    )

    # Línea ideal de perfecta calibración
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Perfect calibration")

    # Confianza promedio observada por bin
    plt.plot(
        bin_centers,
        bin_confs,
        marker="o",
        linewidth=1.5,
        label="Avg confidence"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(
        f"Reliability Diagram\n"
        f"{model_name} | CIFAR-{classes} | Acc={accuracy:.4f} | "
        f"Conf={avg_confidence:.4f} | ECE={ece:.4f}"
    )
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# --------------------------- Main ---------------------------
def main():
    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {model_ckpt_path}")

    print(f"\n>>> Cargando modelo: {model_ckpt_path}\n")
    model = tf.keras.models.load_model(
        model_ckpt_path,
        safe_mode=False,
        compile=False
    )
    print(">>> Recolectando predicciones sobre el test split de CIFAR...")
    y_true, y_prob = collect_predictions(model, val_ds)
    if MODEL_NAME == "vitbase" and TYPES=="teachers":
        all_preds = []
        for imgs, labels in val_ds:
            preds = model(imgs, training=False).numpy()
            all_preds.append(preds)
        y_prob = np.concatenate(all_preds, axis=0)
    stats = compute_calibration_stats(y_true, y_prob, n_bins=N_BINS)

    print("\n===== RESULTADOS =====")
    print(f"Modelo            : {MODEL_NAME}")
    print(f"CIFAR             : {CLASSES}")
    print(f"Checkpoint        : {model_ckpt_path}")
    print(f"Accuracy          : {stats['accuracy']:.6f}")
    print(f"Avg confidence    : {stats['avg_confidence']:.6f}")
    print(f"ECE ({N_BINS} bins)      : {stats['ece']:.6f}")

    # Heurística simple de dirección global
    gap = stats["avg_confidence"] - stats["accuracy"]
    if gap > 0:
        print(f"Global calibration: overconfident (conf - acc = {gap:.6f})")
    elif gap < 0:
        print(f"Global calibration: underconfident (conf - acc = {gap:.6f})")
    else:
        print("Global calibration: perfectly matched on average")

    # Guardar txt
    with open(txt_out_path, "w") as f:
        f.write(f"model={MODEL_NAME}\n")
        f.write(f"classes={CLASSES}\n")
        f.write(f"checkpoint={model_ckpt_path}\n")
        f.write(f"accuracy={stats['accuracy']:.8f}\n")
        f.write(f"avg_confidence={stats['avg_confidence']:.8f}\n")
        f.write(f"ece_bins={N_BINS}\n")
        f.write(f"ece={stats['ece']:.8f}\n")
        f.write(f"confidence_minus_accuracy={gap:.8f}\n")
        # Bins para graficar
        f.write("bin_centers=" + ",".join(f"{v:.6f}" for v in stats['bin_centers']) + "\n")
        f.write("bin_accs="    + ",".join(f"{v:.6f}" for v in stats['bin_accs'])    + "\n")
        f.write("bin_confs="   + ",".join(f"{v:.6f}" for v in stats['bin_confs'])   + "\n")
        f.write("bin_counts="  + ",".join(f"{v}"     for v in stats['bin_counts'])  + "\n")

    # Guardar reliability diagram
    plot_reliability_diagram(
        stats=stats,
        save_path=plot_out_path,
        model_name=MODEL_NAME,
        classes=CLASSES,
        n_bins=N_BINS
    )

    print(f"Resultado guardado en: {txt_out_path}")
    print(f"Diagrama guardado en : {plot_out_path}")


if __name__ == "__main__":
    main()