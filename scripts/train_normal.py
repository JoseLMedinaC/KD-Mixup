import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import argparse
from models.models import (build_mobilenetv2,build_resnet50, build_resnet18) 
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["resnet50", "resnet152v2", "convnexttiny", "convnextlarge", "mobilenetv2", "vitbase", "resnet18"], help="Model to be trained")
args = parser.parse_args()
MODEL = args.model.lower()
# --------------------------- Hipers ---------------------------------
IMG_SIZE  = 224         # se re-escalará dentro del modelo con Lambda
BATCH     = 250
EPOCHS    = 200
INIT_LR   = 1e-3        # igual que la notebook (SGD alta)
DATA_SET  = "cifar100"
NUMBER_CLASS = 100
BASE_DIR =Path(f"checkpoints/students/{DATA_SET}/base_line/{MODEL}/")
CKPT_PATH = Path(BASE_DIR,"student_baseline.keras")
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
# ----------------Check CSV ---------------------
# CSV log setup
csv_path =Path(BASE_DIR,"training_log.csv")
if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss","test_loss" "train_acc", "test_acc"])

class PlotHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epoch = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['accuracy'].append(logs['accuracy'])
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, logs['loss'],logs.get('val_loss'), logs['accuracy'], logs.get('val_accuracy')])
        # Plot and save
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch, self.history['loss'], label='Train Loss')
        plt.plot(self.epoch, self.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch, self.history['accuracy'], label='Train Acc')
        plt.plot(self.epoch, self.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title("Accuracy")
        plt.tight_layout()
        plt.savefig(self.save_path)  # Overwrites each epoch
        plt.close()
# --------------------------- Dataset -------------------------------
def preprocess(img, label, training):
    # ⚠️  NO /255 — EfficientNetV2 espera [0-255]
    if training:
        # Random crop to original size
        pad = int(round(IMG_SIZE/12))
        img = tf.image.resize(img, (IMG_SIZE+pad, IMG_SIZE+pad))
        img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
        img = tf.image.random_flip_left_right(img)
    else:
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, tf.one_hot(label, NUMBER_CLASS)

train_ds, val_ds = tfds.load(DATA_SET, split=["train", "test"], as_supervised=True)

train_ds = (train_ds
            .shuffle(5000)
            .map(lambda x, y: preprocess(x, y, True), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (val_ds
          .map(lambda x, y: preprocess(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH)
          .prefetch(tf.data.AUTOTUNE))

# ----------------- Define Student Model----------------
model, base = globals()[f"build_{MODEL}"](input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUMBER_CLASS, dropout_rate=0.3)
#model, base = build_resnet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUMBER_CLASS , dropout_rate=0.3)
#model = tf.keras.models.load_model(CKPT_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(INIT_LR),#tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# --------------------------- Callbacks ------------------------------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.9, verbose=1,min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(CKPT_PATH, monitor="val_accuracy",
                                       save_best_only=True, verbose=0),
    PlotHistoryCallback(Path(BASE_DIR,"baseline_training_progress.png")),
]
# --------------------------- Fit ------------------------------------
model.summary()
print("\n>>> Entrenando Student\n")
print("Métricas activas:", model.metrics_names)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)
# Save the plot
print(f"\n✅  Mejor modelo guardado en {CKPT_PATH.resolve()}")

