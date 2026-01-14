# train.py - FIXED & COMPATIBLE VERSION

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

print(f"TensorFlow version: {tf.__version__}")

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
MODEL_OUT = "emblemize_net.h5"
CLASSES_OUT = "classes.txt"

# =========================
# DATASET CHECKS
# =========================
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Dataset directory '{DATA_DIR}' not found!")
    print("\nExpected structure:")
    print("dataset/")
    print("  ├── Toyota/")
    print("  ├── Honda/")
    print("  └── OtherBrands/")
    exit(1)

subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
if len(subdirs) == 0:
    print(f"ERROR: No class folders found in '{DATA_DIR}'!")
    exit(1)

print(f"\nFound {len(subdirs)} classes: {subdirs}")

# =========================
# DATA GENERATORS
# =========================
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    zoom_range=0.1,
    validation_split=0.15
)

print("\nLoading training data...")
train_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True,
    class_mode="categorical"
)

print("Loading validation data...")
val_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False,
    class_mode="categorical"
)

# =========================
# SAVE CLASS NAMES
# =========================
class_indices = train_ds.class_indices
class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

with open(CLASSES_OUT, "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"\n✓ Saved {len(class_names)} classes to {CLASSES_OUT}")
print(f"Classes: {class_names}")

# =========================
# BUILD MODEL
# =========================
print("\nBuilding model...")

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# MobileNetV2 preprocessing → [-1, 1]
x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)

x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(train_ds.num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# PHASE 1: TRAIN TOP
# =========================
print("\n" + "=" * 60)
print("PHASE 1: Training classifier head")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# =========================
# PHASE 2: FINE-TUNING
# =========================
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning MobileNetV2")
print("=" * 60)

base.trainable = True
fine_tune_at = 100

for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_epochs = 6
total_epochs = EPOCHS + fine_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=EPOCHS,
    verbose=1
)

# =========================
# SAVE MODEL
# =========================
print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)

try:
    model.save(MODEL_OUT, save_format="h5")
    print(f"✓ Model saved to {MODEL_OUT}")
except Exception as e:
    print(f"Failed to save .h5: {e}")
    fallback = MODEL_OUT.replace(".h5", ".keras")
    model.save(fallback)
    print(f"✓ Model saved to {fallback}")

# =========================
# FINAL METRICS
# =========================
final_train_acc = history_fine.history["accuracy"][-1]
final_val_acc = history_fine.history["val_accuracy"][-1]

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Final Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")
print(f"Model file: {MODEL_OUT}")
print(f"Classes file: {CLASSES_OUT}")
print("\nNext: python app.py")
