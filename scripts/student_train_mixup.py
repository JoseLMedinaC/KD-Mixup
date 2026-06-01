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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from scripts.utils import *
from keras.layers import Lambda
import argparse
#from models.models import build_mobileone_s0
from tensorflow.keras import mixed_precision
from models.models import (build_mobilenetv2,build_resnet50)
mixed_precision.set_global_policy("mixed_float16")

# ---------------------- Argumentos CLI -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--student", type=str, required=True, choices=[
    "resnet50", "mobilenetv2"], help="Modelo student")
parser.add_argument("--teacher", type=str, required=True, choices=[
    "resnet152v2", "convnexttiny", "convnextlarge", "vitbase"
], help="Modelo teacher")
parser.add_argument("--temperature", type=float, default=2.0,
    help="Temperatura de destilación. Usa 0 para T=std (normalización por desviación estándar).")
args = parser.parse_args()
TEACHER_NAME = args.teacher.lower()
STUDENT_NAME = args.student.lower()
# ----------------- Config ---------------------
temperature =  int(args.temperature)
IMG_SIZE = 224
BATCH = 200
EPOCHS = 500
INIT_LR = 1e-3
CLASSES = 100
DATA_SET="cifar"+str(CLASSES)
TEACHER_PATH = f"checkpoints/teachers/{DATA_SET}/best_{TEACHER_NAME}.keras"
STUDENT_PATH = f"checkpoints/students/{DATA_SET}/single_teacher/{STUDENT_NAME}/T{temperature}/distill_{TEACHER_NAME}.keras"
BASE_DIR =Path(f"checkpoints/students/{DATA_SET}/single_teacher/{STUDENT_NAME}/T{temperature}/")
SAVE_PATH = Path(f"checkpoints/students/{DATA_SET}/single_teacher/{STUDENT_NAME}/T{temperature}/distill_{TEACHER_NAME}.keras")

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

def load_student_dataset_nolabels(batch_size=128):
    def preprocess(img, label):
        pad=12
        img = tf.image.resize(img, (IMG_SIZE+pad, IMG_SIZE+pad))
        img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float16)
        return img
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
# ----------------- Teacher ----------------------
teacher = tf.keras.models.load_model(TEACHER_PATH)
teacher = extract_logits_model(teacher, "Teacher1")
teacher.trainable = False
teacher.compile()

# -------------------------------------------------
#student=tf.keras.models.load_model(STUDENT_PATH)
if STUDENT_NAME=="resnet50":
    student, backbone = build_resnet50(num_classes=CLASSES)
else:
    student, backbone = build_mobilenetv2(num_classes=CLASSES)
# Stage 1: train head only
backbone.trainable = False

# ----------------- Distillation Loss ---------------------
def distill_loss(student_logits, teacher_logits):
    if temperature == 0:
        # Rescaled logits approach (Choi et al. 2023)
        std_s = tf.math.reduce_std(student_logits, axis=1, keepdims=True)
        std_t = tf.math.reduce_std(teacher_logits, axis=1, keepdims=True)
        student_scaled = student_logits / (std_s + 1e-8)
        teacher_scaled = teacher_logits / (std_t + 1e-8)
        teacher_probs = tf.nn.softmax(teacher_scaled)
        student_log_probs = tf.nn.log_softmax(student_scaled)
        kl = tf.reduce_mean(tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-7) - student_log_probs), axis=1
        ))
        return kl
    else:
        # Original approach con temperatura fija
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)
        student_log_probs = tf.nn.log_softmax(student_logits / temperature)
        kl = tf.reduce_mean(tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-7) - student_log_probs), axis=1
        ))
        return kl * (temperature ** 2)

# --------------------------------------------------------

# ----------------- Training Step -------------------------


@tf.function(input_signature=[tf.TensorSpec(shape=(BATCH, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float16)])
def train_step(x):
    indices = tf.random.shuffle(tf.range(BATCH))
    λ = tf.random.uniform([BATCH, 1, 1, 1],dtype=tf.float16)
    x = λ * x + (1 - λ) * tf.gather(x, indices)
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
        loss = distill_loss(student_logits, teacher_logits)
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


#---------------------Traing top Classifier----------------------
train_ds = load_student_dataset(BATCH)
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.9, verbose=1,min_lr=1e-5)]
student.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=["accuracy"]) #,tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=0.9)
student.fit(train_ds,epochs=10,callbacks=callbacks)
#------------------------------------------
#-------------Train Backbone---------
optimizer = tf.keras.optimizers.Adam(INIT_LR)
train_ds= load_student_dataset_nolabels(BATCH)
test_ds = load_test_dataset(BATCH)
train_losses = []
test_accuracies = []
backbone.trainable = True
max_acc=0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    segments = read_lr_schedule("./config/schedule_lr_ST.csv")
    INIT_LR = get_lr_from_segments(epoch, segments)
    optimizer.learning_rate.assign(INIT_LR)
    epoch_losses = []
    progress_bar = tqdm(train_ds, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for x in progress_bar:
        loss = train_step(x)
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

