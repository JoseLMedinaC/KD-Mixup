#!/usr/bin/env python3
"""
Train a student model on CIFAR 10-100 using soft logits distillation and mixup augmentation.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from keras import config as keras_config
from tqdm import tqdm
import os
import csv
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from scripts.distill import *
from keras.layers import Lambda
import argparse
#from models.models import build_mobileone_s0
from tensorflow.keras import mixed_precision
from models.models import (build_mobilenetv2,build_resnet50, build_resnet18)
mixed_precision.set_global_policy("mixed_float16")

# ---------------------- Argumentos CLI -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--student", type=str, required=True, choices=["resnet50", "mobilenetv2", "resnet18"], help="student model")
parser.add_argument("--teacher", type=str, required=True, choices=["resnet152v2", "convnexttiny", "convnextlarge", "vitbase","resnet18"], help="teacher model")
parser.add_argument("--logits", action=argparse.BooleanOptionalAction, default=True, help="Extract logits from teacher, default True --no-logits for False")
parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation, 0 for T=std (norm by standar deviation).")
parser.add_argument("--method", type=str, default="mixup", choices=["mixup", "lskd", "rld"], help="KD method")
parser.add_argument("--alpha", type=float, default=1, help="Alpha for Beta(alpha, alpha)")
parser.add_argument("--partial", type=float, default=1, help="Partial mixup 0-1")
args = parser.parse_args()
#Selecting Loss Function
LOSSES = {
    "rld": rld_loss,
    "lskd": lskd_loss,
    "mixup": mixup_loss,
}
STUDENT = {
    "mobilenetv2": build_mobilenetv2,
    "resnet50": build_resnet50,
    "resnet18": build_resnet18,
}
METHOD =args.method
TEACHER_NAME = args.teacher.lower()
STUDENT_NAME = args.student.lower()
LOGITS = args.logits
distill_loss = LOSSES[METHOD]
build_student = STUDENT[STUDENT_NAME]
# ----------------- Config ---------------------
TEMPERATURE =  int(args.temperature)
IMG_SIZE = 224
BATCH = 200
EPOCHS = 300
WARMUP_EPOCHS = 10
INIT_LR = 1e-3
CLASSES = 100
DATA_SET="cifar"+str(CLASSES)
TEACHER_PATH = f"checkpoints/teachers/{DATA_SET}/best_{TEACHER_NAME}.keras"
STUDENT_PATH = f"checkpoints/students/{DATA_SET}/{METHOD}/single_teacher/{STUDENT_NAME}/T{TEMPERATURE}/distill_{TEACHER_NAME}.keras"
BASE_DIR =Path(f"checkpoints/students/{DATA_SET}/{METHOD}/single_teacher/{STUDENT_NAME}/T{TEMPERATURE}/")
SAVE_PATH = Path(STUDENT_PATH)

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
    print(str(g)," Set memory growth")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
keras_config.enable_unsafe_deserialization()
csv_path =Path(BASE_DIR,f"training_log_{TEACHER_NAME}.csv")
if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","lr", "train_loss", "test_acc"])
# -----------------------------------------------

# ----------------- Load Dataset ----------------
def load_student_dataset(batch_size=128):
    def preprocess(img, label):
        pad=12
        img = tf.image.resize(img, (IMG_SIZE+pad, IMG_SIZE+pad))
        img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float16)
        label = tf.one_hot(label, depth=CLASSES)
        label = tf.cast(label, tf.float16)
        return img, label
    ds_raw = tfds.load(DATA_SET, split="train", as_supervised=True)
    ds = ds_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(2500).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def load_test_dataset(batch_size=128):
    def preprocess(image, label):
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float16)
        label = tf.one_hot(label, depth=CLASSES)
        label = tf.cast(label, tf.float16)
        return image, label
    test_ds = tfds.load(DATA_SET, split="test", as_supervised=True)
    return test_ds.map(preprocess).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# ------------------------------------------------
def extract_logits_model(model, new_input_name="renamed_input"):
    """Devuelve submodelo hasta la capa 'logits' con input renombrado."""
    logits_output = model.get_layer("logits").output
    logits_model = Model(inputs=model.input, outputs=logits_output)
    old_input = logits_model.input
    if isinstance(old_input, list):
        old_input = old_input[0]
    new_input = Input(shape=old_input.shape[1:], name=new_input_name, dtype="float16")
    new_output = logits_model(new_input)
    new_output = Lambda(lambda x: tf.cast(x, tf.float16), dtype="float16")(new_output)
    return Model(inputs=new_input, outputs=new_output)
# ----------------- Distillation Loss ---------------------


# --------------------------------------------------------
def sample_beta(alpha, shape):
    x = tf.random.gamma(shape, alpha=alpha, dtype=tf.float32)
    y = tf.random.gamma(shape, alpha=alpha, dtype=tf.float32)
    return tf.cast(x / (x + y + 1e-8), tf.float16)

# ----------------- Training Step -------------------------
@tf.function(input_signature=[tf.TensorSpec(shape=(BATCH, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float16), tf.TensorSpec(shape=(BATCH, CLASSES), dtype=tf.float16),])
def train_step(x,label):
    if METHOD=="mixup":
        _ALPHA=1
        _PARTIAL=1
        indices = tf.random.shuffle(tf.range(BATCH))
        lam  = sample_beta(_ALPHA, [BATCH, 1])                              # [BATCH,1]
        mask = tf.cast(tf.random.uniform([BATCH, 1]) < _PARTIAL, tf.float16) # [BATCH,1]
        lam  = mask * lam + (1 - mask) * tf.ones_like(lam)                   # [BATCH,1]
        # para imágenes: reshape a [BATCH,1,1,1]
        lam_img = tf.reshape(lam, [BATCH, 1, 1, 1])
        x = lam_img * x + (1 - lam_img) * tf.gather(x, indices)
        # para labels: ya está en [BATCH,1], hace broadcast directo con [BATCH,CLASSES]
        label = lam * label + (1 - lam) * tf.gather(label, indices)
    #Normalizacion
    if TEACHER_NAME == "vitbase":
        x_teacher = tf.cast(x, tf.float32) / 255.0
        x_teacher = (x_teacher - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    else:
        x_teacher = x
    teacher_logits = teacher(x_teacher, training=False)
    teacher_logits = tf.stop_gradient(teacher_logits)
    
    with tf.GradientTape() as tape:
        student_logits = student(x, training=True)
        student_logits=tf.cast(student_logits, tf.float16)
        loss = distill_loss(student_logits, teacher_logits,label,TEMPERATURE)
    grads = tape.gradient(loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return loss
# --------------------------------------------------------

@tf.function(input_signature=[
    tf.TensorSpec(shape=(BATCH, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float16),
    tf.TensorSpec(shape=(BATCH, CLASSES), dtype=tf.float16),   # o float32, pero consistente
])
def test_step(x, y_true):
    logits = student(x, training=False)
    preds = tf.argmax(logits, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    acc = tf.cast(tf.equal(preds, y_true), tf.float16)
    return tf.reduce_mean(acc)

# ----------------- Training Loop -------------------------
def save_training_plot(losses, accuracies, filename="student_training_progress.png"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Test Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ===================== LR SCHEDULE =====================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # ---- warmup ----
        warmup_lr = self.base_lr * (step / self.warmup_steps)

        # ---- cosine decay ----
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)

        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + tf.cos(math.pi * progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

# ----------------- Teacher ----------------------
teacher = tf.keras.models.load_model(TEACHER_PATH)
if LOGITS:
    teacher = extract_logits_model(teacher, "Teacher1")
teacher.trainable = False

# -------------------------------------------------
student, backbone = build_student(input_shape=(IMG_SIZE, IMG_SIZE, 3),num_classes=CLASSES, weights=None)

#---------------------Traing top Classifier----------------------
train_ds= load_student_dataset(BATCH)

#-------------Train Backbone---------
optimizer = tf.keras.optimizers.Adam(INIT_LR)
ds_raw = tfds.load(DATA_SET, split="train", as_supervised=True)
N = tf.data.experimental.cardinality(ds_raw).numpy()
steps_per_epoch     = N // BATCH
total_steps         = EPOCHS * steps_per_epoch
warmup_steps        = WARMUP_EPOCHS * steps_per_epoch
lr_schedule         = WarmupCosineDecay(
                            base_lr=3e-4,
                            warmup_steps=warmup_steps,
                            total_steps=total_steps,
                            min_lr=1e-5,
                        )
optimizer           = tf.keras.optimizers.AdamW(
                            learning_rate=lr_schedule,
                            weight_decay=1e-4,
                        )
test_ds = load_test_dataset(BATCH)
train_losses = []
test_accuracies = []
backbone.trainable = True
max_acc=0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    epoch_losses = []
    progress_bar = tqdm(train_ds, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for x, label in progress_bar:
        loss = train_step(x,label)
        epoch_losses.append(loss.numpy())
    avg_loss = np.mean(epoch_losses).item()
    train_losses.append(avg_loss)
    accs = [float(test_step(xb, yb).numpy()) for xb, yb in test_ds]
    avg_acc = float(np.mean(accs))
    test_accuracies.append(avg_acc)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1,INIT_LR, avg_loss, avg_acc])
    if avg_acc>max_acc:
        student.save(SAVE_PATH)
        print(f"\n Best Student model saved to {SAVE_PATH}")
        max_acc=avg_acc
    print(f"Test Accuracy: {avg_acc:.4f}")
    save_training_plot(train_losses, test_accuracies,filename=Path(BASE_DIR,f"student_training_single_{TEACHER_NAME}.png"))

# ----------------- Save Model -------------------------

