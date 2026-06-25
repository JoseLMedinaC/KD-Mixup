"""
Stage 4 — Teacher vs Student on Interpolated Inputs
=====================================================
For each teacher-student pair measures on the TRAINING SET:

1. L_NL of the student
   KL( λ·p_s(xi) + (1-λ)·p_s(xj)  ||  p_s(x̃) )
   → compare with teacher L_NL from Stage 1
   → did Mixup make the student more linear than its teacher?

2. Entropy comparison H[p_t(x̃)] vs H[p_s(x̃)] per λ bin
   → is the student more uncertain than the teacher in ambiguous regions?

3. Teacher-student alignment on interpolated inputs
   KL( p_t(x̃) || p_s(x̃) ) per λ bin
   → how well does the student follow the teacher on x̃?

Outputs
-------
results/stage4/<teacher_name>/
    comparison_by_lambda.csv
    summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ───────────────────────── CLI ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--teacher", type=str, required=True,
                    choices=["resnet152v2", "convnexttiny", "convnextlarge", "vitbase"])
parser.add_argument("--student", type=str, required=True,
                    choices=["resnet50", "mobilenetv2"])
parser.add_argument("--classes",     type=int, default=100)
parser.add_argument("--batch",       type=int, default=256)
parser.add_argument("--n_batches",   type=int, default=100)
parser.add_argument("--img_size",    type=int, default=224)
parser.add_argument("--lambda_bins", type=int, default=20)
parser.add_argument("--out_dir",     type=str, default="results/stage4")
args = parser.parse_args()

IMG_SIZE     = args.img_size
BATCH        = args.batch
CLASSES      = args.classes
TEACHER_NAME = args.teacher.lower()
STUDENT_NAME = args.student.lower()
DATASET      = f"cifar{CLASSES}"
TEMPERATURE  = 2
TEACHER_PATH = ROOT / f"checkpoints/teachers/cifar{CLASSES}/best_{TEACHER_NAME}.keras"
#STUDENT_PATH = ROOT / f"checkpoints/students/{DATASET}/base_line/distill_{TEACHER_NAME}.keras"
STUDENT_PATH = ROOT / f"checkpoints/students/{DATASET}/single_teacher/{STUDENT_NAME}/T{TEMPERATURE}/distill_{TEACHER_NAME}.keras"
OUT_DIR      = ROOT / args.out_dir / STUDENT_NAME /TEACHER_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Stage 4 — Teacher vs Student Analysis")
print(f"  Teacher : {TEACHER_NAME}")
print(f"  Dataset : {DATASET}")
print(f"  Batches : {args.n_batches}  x  batch_size={BATCH}")
print(f"{'='*60}\n")

# ───────────────────────── Helpers ─────────────────────────

def extract_logits_model(model: tf.keras.Model,
                         new_input_name: str = "renamed_input") -> tf.keras.Model:
    """Extract logits sub-model from teacher (has explicit 'logits' layer)."""
    logits_output = model.get_layer("logits").output
    logits_model  = Model(inputs=model.input, outputs=logits_output)
    old_input     = logits_model.input
    if isinstance(old_input, list):
        old_input = old_input[0]
    new_input  = Input(shape=old_input.shape[1:],
                       name=new_input_name, dtype="float16")
    new_output = logits_model(new_input)
    new_output = Lambda(lambda x: tf.cast(x, tf.float16),
                        dtype="float16")(new_output)
    return Model(inputs=new_input, outputs=new_output)


def entropy(p: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    """Shannon entropy per sample [B,K] → [B]."""
    p = tf.cast(p, tf.float32) + eps
    return -tf.reduce_sum(p * tf.math.log(p), axis=-1)


def kl_div(p: tf.Tensor, q: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    """KL(p || q) per sample [B,K] → [B]."""
    p = tf.cast(p, tf.float32) + eps
    q = tf.cast(q, tf.float32) + eps
    return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)


def softmax_f32(logits: tf.Tensor) -> tf.Tensor:
    return tf.nn.softmax(tf.cast(logits, tf.float32))


# ───────────────────────── Dataset ─────────────────────────

def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float16)
    label = tf.one_hot(label, depth=CLASSES)
    label = tf.cast(label, tf.float16)
    return image, label

print("Loading training dataset …")
train_ds = (
    tfds.load(DATASET, split="train", as_supervised=True)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# ───────────────────────── Models ─────────────────────────

print(f"Loading teacher from {TEACHER_PATH} …")
teacher_full   = tf.keras.models.load_model(str(TEACHER_PATH), safe_mode=False)
teacher_logits = extract_logits_model(teacher_full, f"teacher_{TEACHER_NAME}")
teacher_logits.trainable = False
teacher_logits.compile()
print("Teacher loaded.")

print(f"Loading student from {STUDENT_PATH} …")
# Student outputs logits directly — no extraction needed
student = tf.keras.models.load_model(str(STUDENT_PATH), safe_mode=False)
student.trainable = False
student.compile()
print("Student loaded.\n")

# ───────────────────────── Evaluation loop ─────────────────────────
@tf.function
def normalize_for_teacher(x: tf.Tensor) -> tf.Tensor:
    """Apply ImageNet normalization only for ViT, passthrough otherwise."""
    if TEACHER_NAME == "vitbase":
        x = tf.cast(x, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        return (x - mean) / std
    return tf.cast(x, tf.float32)

@tf.function
def compute_batch(x_i: tf.Tensor, x_j: tf.Tensor, lam: tf.Tensor):
    """
    For each pair (x_i, x_j) with mixing coefficient λ computes:

    Teacher side:
        p_ti, p_tj    = softmax(teacher(xi)), softmax(teacher(xj))
        p_t_mix       = softmax(teacher(x̃))
        p_t_linear    = λ·p_ti + (1-λ)·p_tj
        L_NL_t        = KL(p_t_linear || p_t_mix)
        H_t           = H[p_t_mix]

    Student side:
        p_si, p_sj    = softmax(student(xi)), softmax(student(xj))
        p_s_mix       = softmax(student(x̃))
        p_s_linear    = λ·p_si + (1-λ)·p_sj
        L_NL_s        = KL(p_s_linear || p_s_mix)
        H_s           = H[p_s_mix]

    Alignment:
        KL_ts         = KL(p_t_mix || p_s_mix)   teacher→student on x̃
        KL_ts_real    = KL(p_ti    || p_si)        teacher→student on real xi

    Returns all per-sample, shape [B].
    """
    lam_img  = tf.reshape(lam, [-1, 1, 1, 1])
    lam_prob = tf.reshape(tf.cast(lam, tf.float32), [-1, 1])

    x_mix = lam_img * x_i + (1.0 - lam_img) * x_j


    # ── Teacher (normalized if needed) ──
    x_i_t   = normalize_for_teacher(x_i)
    x_j_t   = normalize_for_teacher(x_j)
    x_mix_t = normalize_for_teacher(x_mix)
    # ── Teacher ──


    p_ti    = softmax_f32(teacher_logits(x_i_t,   training=False))
    p_tj    = softmax_f32(teacher_logits(x_j_t,   training=False))
    p_t_mix = softmax_f32(teacher_logits(x_mix_t, training=False))
    p_t_lin = lam_prob * p_ti + (1.0 - lam_prob) * p_tj

    L_NL_t = kl_div(p_t_lin, p_t_mix)
    H_t    = entropy(p_t_mix)

    # ── Student ──
    p_si    = softmax_f32(student(x_i,   training=False))
    p_sj    = softmax_f32(student(x_j,   training=False))
    p_s_mix = softmax_f32(student(x_mix, training=False))
    p_s_lin = lam_prob * p_si + (1.0 - lam_prob) * p_sj

    L_NL_s = kl_div(p_s_lin, p_s_mix)
    H_s    = entropy(p_s_mix)

    # ── Alignment ──
    KL_ts      = kl_div(p_t_mix, p_s_mix)   # on interpolated input
    KL_ts_real = kl_div(p_ti,    p_si)       # on real input xi

    lam_out = tf.cast(lam, tf.float32)
    return L_NL_t, L_NL_s, H_t, H_s, KL_ts, KL_ts_real, lam_out


accum = {k: [] for k in
         ["L_NL_t","L_NL_s","H_t","H_s","KL_ts","KL_ts_real","lam"]}

print("Running evaluation …\n")
for batch_idx, (x_batch, _) in enumerate(train_ds):
    if batch_idx >= args.n_batches:
        break

    indices = tf.random.shuffle(tf.range(BATCH))
    x_i = x_batch
    x_j = tf.gather(x_batch, indices)
    lam = tf.random.uniform([BATCH], dtype=tf.float16)

    L_NL_t, L_NL_s, H_t, H_s, KL_ts, KL_ts_real, lam_vals = \
        compute_batch(x_i, x_j, lam)

    for k, v in zip(accum.keys(),
                    [L_NL_t, L_NL_s, H_t, H_s, KL_ts, KL_ts_real, lam_vals]):
        accum[k].append(v.numpy())

    if (batch_idx + 1) % 10 == 0:
        print(f"  Batch {batch_idx+1:>4}/{args.n_batches}  "
              f"L_NL_t={L_NL_t.numpy().mean():.3f}  "
              f"L_NL_s={L_NL_s.numpy().mean():.3f}  "
              f"H_t={H_t.numpy().mean():.3f}  "
              f"H_s={H_s.numpy().mean():.3f}  "
              f"KL_ts={KL_ts.numpy().mean():.3f}")

for k in accum:
    accum[k] = np.concatenate(accum[k])

print(f"\nTotal pairs evaluated: {len(accum['lam']):,}")

# ───────────────────────── Binning ─────────────────────────

bins        = np.linspace(0.0, 1.0, args.lambda_bins + 1)
bin_idx     = np.clip(np.digitize(accum["lam"], bins) - 1, 0, args.lambda_bins - 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

rows = []
for b in range(args.lambda_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue

    def m(k): return float(accum[k][mask].mean())

    # linearity gain: how much more linear is student vs teacher
    # positive = student is MORE linear (lower L_NL) than teacher
    lin_gain = m("L_NL_t") - m("L_NL_s")

    # entropy gap: student entropy minus teacher entropy on x̃
    # positive = student is MORE uncertain than teacher on mixed inputs
    ent_gap = m("H_s") - m("H_t")

    rows.append({
        "lambda_center":  bin_centers[b],
        "L_NL_teacher":   m("L_NL_t"),
        "L_NL_student":   m("L_NL_s"),
        "linearity_gain": lin_gain,          # L_NL_t - L_NL_s
        "H_teacher":      m("H_t"),
        "H_student":      m("H_s"),
        "entropy_gap":    ent_gap,           # H_s - H_t
        "KL_ts_mixed":    m("KL_ts"),        # alignment on x̃
        "KL_ts_real":     m("KL_ts_real"),   # alignment on real x
        "count":          int(mask.sum()),
    })

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "comparison_by_lambda.csv"
df.to_csv(csv_path, index=False)
print(f"\nComparison table saved → {csv_path}")

# ───────────────────────── Summary ─────────────────────────

mid = (accum["lam"] > 0.45) & (accum["lam"] < 0.55)

summary = {
    "teacher":                    TEACHER_NAME,
    "dataset":                    DATASET,
    "n_pairs":                    int(len(accum["lam"])),

    # Non-linearity
    "global_L_NL_teacher":        float(accum["L_NL_t"].mean()),
    "global_L_NL_student":        float(accum["L_NL_s"].mean()),
    "global_linearity_gain":      float((accum["L_NL_t"] - accum["L_NL_s"]).mean()),
    "peak_L_NL_teacher":          float(accum["L_NL_t"][mid].mean()),
    "peak_L_NL_student":          float(accum["L_NL_s"][mid].mean()),
    "peak_linearity_gain":        float((accum["L_NL_t"] - accum["L_NL_s"])[mid].mean()),

    # Entropy
    "global_H_teacher":           float(accum["H_t"].mean()),
    "global_H_student":           float(accum["H_s"].mean()),
    "global_entropy_gap":         float((accum["H_s"] - accum["H_t"]).mean()),

    # Alignment
    "global_KL_ts_mixed":         float(accum["KL_ts"].mean()),
    "global_KL_ts_real":          float(accum["KL_ts_real"].mean()),
    "alignment_degradation":      float(
        accum["KL_ts"].mean() - accum["KL_ts_real"].mean()
    ),  # how much worse is alignment on mixed vs real inputs
}

json_path = OUT_DIR / "summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved           → {json_path}")

print("\n── Summary ──────────────────────────────────────────────")
for k, v in summary.items():
    print(f"  {k:<35} {v}")
print("─────────────────────────────────────────────────────────\n")
print("Stage 4 complete.")