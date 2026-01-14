# debug_model.py - Comprehensive Model Debugging Script
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 70)
print("EmblemizeAI - Debugging Script")
print("=" * 70)

# Step 1: Check files
print("\n1. Checking Required Files...")
print("-" * 70)

files_status = {}
critical_files = {
    "emblemize_net.h5": "Trained model (H5 format)",
    "emblemize_model.tflite": "TFLite model (optional)",
    "classes.txt": "Class names file",
    "dataset/": "Training dataset folder"
}

all_good = True
for file, desc in critical_files.items():
    exists = os.path.exists(file)
    files_status[file] = exists
    status = "✓" if exists else "✗"

    if exists:
        if os.path.isfile(file):
            size = os.path.getsize(file) / (1024 * 1024)
            print(f"  {status} {file:25s} - {desc} ({size:.2f} MB)")
        else:
            print(f"  {status} {file:25s} - {desc} (folder)")
    else:
        print(f"  {status} {file:25s} - {desc} (MISSING!)")
        all_good = False

if not files_status.get("emblemize_net.h5"):
    print("\n  ❌ CRITICAL: Model file not found!")
    print("  You need to train the model first: python train.py")
    sys.exit(1)

# Step 2: Check TensorFlow
print("\n2. Checking TensorFlow Installation...")
print("-" * 70)

try:
    import tensorflow as tf

    print(f"  ✓ TensorFlow version: {tf.__version__}")
    print(f"  ✓ Keras version: {tf.keras.__version__}")

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ GPU Available: {len(gpus)} device(s)")
    else:
        print(f"  ⚠ No GPU detected (using CPU)")

except ImportError:
    print("  ✗ TensorFlow not installed!")
    sys.exit(1)

# Step 3: Load and inspect model
print("\n3. Loading and Inspecting Model...")
print("-" * 70)

try:
    model = tf.keras.models.load_model("emblemize_net.h5", compile=False)
    print("  ✓ Model loaded successfully")

    print(f"\n  Model Architecture:")
    print(f"    Input shape:  {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")
    print(f"    Total params: {model.count_params():,}")
    print(f"    Layers:       {len(model.layers)}")

    num_classes = model.output_shape[-1]
    print(f"\n  ✓ Model expects {num_classes} classes")

except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    sys.exit(1)

# Step 4: Check classes.txt
print("\n4. Checking Class Names...")
print("-" * 70)

try:
    with open("classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"  ✓ Loaded {len(class_names)} class names")
    print(f"  Classes: {class_names}")

    if len(class_names) != num_classes:
        print(f"\n  ⚠ WARNING: Mismatch detected!")
        print(f"    classes.txt has {len(class_names)} classes")
        print(f"    Model expects {num_classes} classes")
        print(f"  This will cause incorrect predictions!")
    else:
        print(f"  ✓ Class count matches model output")

except Exception as e:
    print(f"  ✗ Error reading classes.txt: {e}")
    class_names = [f"Class_{i}" for i in range(num_classes)]

# Step 5: Check dataset
print("\n5. Checking Dataset...")
print("-" * 70)

if os.path.exists("dataset"):
    brand_folders = [d for d in os.listdir("dataset")
                     if os.path.isdir(os.path.join("dataset", d))]

    if brand_folders:
        print(f"  ✓ Found {len(brand_folders)} brand folders")

        print(f"\n  Dataset brands vs Model classes:")
        dataset_brands = set(brand_folders)
        model_brands = set(class_names)

        # Check match
        if dataset_brands == model_brands:
            print(f"  ✓ Perfect match!")
        else:
            missing_in_model = dataset_brands - model_brands
            missing_in_dataset = model_brands - dataset_brands

            if missing_in_model:
                print(f"  ⚠ Brands in dataset but NOT in model: {missing_in_model}")
            if missing_in_dataset:
                print(f"  ⚠ Brands in model but NOT in dataset: {missing_in_dataset}")

        # Show image counts
        print(f"\n  Images per brand:")
        for brand in sorted(brand_folders):
            path = os.path.join("dataset", brand)
            images = [f for f in os.listdir(path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"    {brand:20s}: {len(images):4d} images")
    else:
        print(f"  ✗ No brand folders found in dataset/")
else:
    print(f"  ✗ dataset/ folder not found")

# Step 6: Test model with sample predictions
print("\n6. Testing Model Predictions...")
print("-" * 70)


def preprocess_image(img_path):
    """Preprocess image exactly like the app does"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# Find some test images
test_images = []
if os.path.exists("dataset"):
    for brand in os.listdir("dataset"):
        brand_path = os.path.join("dataset", brand)
        if os.path.isdir(brand_path):
            images = [os.path.join(brand_path, f) for f in os.listdir(brand_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_images.append((brand, images[0]))  # Take first image
                if len(test_images) >= 5:  # Test 5 brands
                    break

if test_images:
    print(f"  Testing with {len(test_images)} sample images from dataset...\n")

    correct = 0
    total = len(test_images)

    for true_brand, img_path in test_images:
        try:
            # Preprocess and predict
            x = preprocess_image(img_path)
            preds = model.predict(x, verbose=0)[0]

            # Get top 3 predictions
            top3_idx = np.argsort(preds)[-3:][::-1]

            predicted_brand = class_names[top3_idx[0]]
            confidence = preds[top3_idx[0]] * 100

            # Check if correct
            is_correct = (predicted_brand.lower() == true_brand.lower())
            if is_correct:
                correct += 1

            status = "✓" if is_correct else "✗"
            print(f"  {status} True: {true_brand:15s} | Predicted: {predicted_brand:15s} ({confidence:.1f}%)")

            # Show top 3
            print(f"      Top 3: ", end="")
            for idx in top3_idx:
                print(f"{class_names[idx]} ({preds[idx] * 100:.1f}%), ", end="")
            print()

        except Exception as e:
            print(f"  ✗ Error testing {img_path}: {e}")

    accuracy = (correct / total) * 100
    print(f"\n  Sample Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    if accuracy < 50:
        print(f"\n  ❌ LOW ACCURACY DETECTED!")
        print(f"  Possible causes:")
        print(f"    1. Model not trained enough (try more epochs)")
        print(f"    2. Dataset quality issues (mixed/mislabeled images)")
        print(f"    3. Model-dataset mismatch (wrong classes.txt)")
        print(f"    4. Preprocessing mismatch between train and predict")
    elif accuracy < 80:
        print(f"\n  ⚠ MODERATE ACCURACY - Could be better")
        print(f"  Consider retraining with more epochs or data augmentation")
    else:
        print(f"\n  ✓ GOOD ACCURACY on sample images!")

else:
    print(f"  ⚠ No test images found in dataset/")

# Step 7: Check preprocessing
print("\n7. Checking Image Preprocessing...")
print("-" * 70)

print(f"  Expected input:")
print(f"    - Image size: 224x224")
print(f"    - Color mode: RGB")
print(f"    - Value range: 0.0 to 1.0 (normalized)")
print(f"    - Shape: (1, 224, 224, 3)")

if test_images:
    img_path = test_images[0][1]
    x = preprocess_image(img_path)
    print(f"\n  ✓ Sample preprocessing check:")
    print(f"    Shape: {x.shape}")
    print(f"    Data type: {x.dtype}")
    print(f"    Value range: {x.min():.3f} to {x.max():.3f}")

    if x.shape != (1, 224, 224, 3):
        print(f"  ❌ Shape mismatch!")
    if x.dtype != np.float32:
        print(f"  ⚠ Data type should be float32")
    if x.min() < 0 or x.max() > 1.1:
        print(f"  ❌ Value range incorrect!")

# Step 8: Summary and recommendations
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

issues_found = []

# Check critical issues
if len(class_names) != num_classes:
    issues_found.append("❌ CRITICAL: classes.txt doesn't match model output!")

if test_images and accuracy < 50:
    issues_found.append("❌ CRITICAL: Very low accuracy on test images!")

if not files_status.get("dataset/"):
    issues_found.append("⚠ Dataset folder not found")

if issues_found:
    print("\n🔴 ISSUES DETECTED:")
    for issue in issues_found:
        print(f"  {issue}")

    print("\n📋 RECOMMENDED FIXES:")

    if len(class_names) != num_classes:
        print(f"\n  1. Fix classes.txt:")
        print(f"     The model expects {num_classes} classes, but classes.txt has {len(class_names)}")
        print(f"     Solution: Retrain the model with current dataset")
        print(f"     Command: python train.py")

    if test_images and accuracy < 50:
        print(f"\n  2. Retrain the model:")
        print(f"     Low accuracy suggests the model isn't properly trained")
        print(f"     Solution: Train for more epochs with better data")
        print(f"     Command: python train.py")

    print(f"\n  3. Verify your dataset:")
    print(f"     - Check if images are correctly labeled")
    print(f"     - Remove corrupted or mislabeled images")
    print(f"     - Ensure at least 50+ images per brand")

else:
    print("\n✅ No major issues detected!")
    print("\nIf predictions are still wrong, check:")
    print("  1. Are you testing with clear, well-lit car logo images?")
    print("  2. Is the logo centered and visible in the image?")
    print("  3. Are you testing brands that exist in the training data?")

print("\n" + "=" * 70)
print(f"For better recognition, ensure:")
print(f"  • Images show clear, front-facing car logos/emblems")
print(f"  • Good lighting and resolution")
print(f"  • Logo takes up significant portion of image")
print(f"  • Testing brands that model was trained on")
print("=" * 70)