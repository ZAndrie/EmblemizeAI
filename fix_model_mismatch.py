# fix_model_mismatch.py - Diagnose and fix the class mismatch issue
import os
import sys
import numpy as np
from PIL import Image

print("=" * 70)
print("FIXING MODEL-DATASET MISMATCH")
print("=" * 70)

# Step 1: Check what's in your current setup
print("\n1. Current Setup Check")
print("-" * 70)

# Check classes.txt
if os.path.exists("classes.txt"):
    with open("classes.txt", "r") as f:
        current_classes = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✓ classes.txt has {len(current_classes)} classes:")
    print(f"  {current_classes}")
else:
    print("✗ classes.txt not found!")
    sys.exit(1)

# Check dataset folder
if os.path.exists("dataset"):
    dataset_brands = sorted([d for d in os.listdir("dataset")
                             if os.path.isdir(os.path.join("dataset", d))])
    print(f"\n✓ dataset/ folder has {len(dataset_brands)} brands:")
    print(f"  {dataset_brands}")
else:
    print("✗ dataset/ folder not found!")
    sys.exit(1)

# Check model
try:
    import tensorflow as tf

    if os.path.exists("emblemize_net.h5"):
        model = tf.keras.models.load_model("emblemize_net.h5", compile=False)
        model_classes = model.output_shape[-1]
        print(f"\n✓ Model expects {model_classes} classes")
    else:
        print("✗ emblemize_net.h5 not found!")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Step 2: Identify the mismatch
print("\n" + "=" * 70)
print("2. MISMATCH ANALYSIS")
print("=" * 70)

problems = []

if len(current_classes) != len(dataset_brands):
    problems.append(f"classes.txt ({len(current_classes)}) ≠ dataset brands ({len(dataset_brands)})")

if len(current_classes) != model_classes:
    problems.append(f"classes.txt ({len(current_classes)}) ≠ model output ({model_classes})")

if len(dataset_brands) != model_classes:
    problems.append(f"dataset brands ({len(dataset_brands)}) ≠ model output ({model_classes})")

if set(current_classes) != set(dataset_brands):
    missing_in_classes = set(dataset_brands) - set(current_classes)
    missing_in_dataset = set(current_classes) - set(dataset_brands)
    if missing_in_classes:
        problems.append(f"Brands in dataset but NOT in classes.txt: {missing_in_classes}")
    if missing_in_dataset:
        problems.append(f"Brands in classes.txt but NOT in dataset: {missing_in_dataset}")

if problems:
    print("\n⚠️  PROBLEMS DETECTED:")
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. {problem}")

    print("\n" + "=" * 70)
    print("3. THE FIX")
    print("=" * 70)

    print("\n🔴 YOUR MODEL WAS TRAINED ON DIFFERENT CLASSES!")
    print("\nThis is why predictions are wrong. The model learned to predict")
    print("classes that don't match your current dataset/classes.txt.")

    print("\n📋 SOLUTION: Delete model and retrain")
    print("\nRun these commands:")
    print("  1. rm emblemize_net.h5")
    print("  2. python train.py")

    print("\nThis will:")
    print("  • Create a fresh model matching your current dataset")
    print("  • Generate correct classes.txt automatically")
    print("  • Train on all 8 brands in your dataset folder")

    # Check training data quality
    print("\n" + "=" * 70)
    print("4. DATASET QUALITY CHECK")
    print("=" * 70)

    print("\nBefore retraining, verify your dataset:")
    for brand in dataset_brands[:8]:
        path = os.path.join("dataset", brand)
        images = [f for f in os.listdir(path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        status = "✓" if len(images) >= 50 else "⚠️"
        print(f"  {status} {brand:15s}: {len(images):4d} images")

    print("\n💡 Recommendations:")
    print("  • Each brand should have at least 50 images")
    print("  • Images should be clear photos of car logos/emblems")
    print("  • Manually check 5-10 images per brand to verify quality")

else:
    print("\n✅ No mismatch detected!")
    print("\nBut predictions are still wrong? Let's test the model...")

    print("\n" + "=" * 70)
    print("3. TESTING MODEL ON YOUR DATA")
    print("=" * 70)


    def preprocess_image(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)


    # Test 3 images per brand
    total_correct = 0
    total_tested = 0

    for brand in dataset_brands[:8]:
        path = os.path.join("dataset", brand)
        images = [f for f in os.listdir(path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]

        brand_correct = 0
        for img_name in images:
            img_path = os.path.join(path, img_name)
            try:
                x = preprocess_image(img_path)
                preds = model.predict(x, verbose=0)[0]
                predicted_idx = int(np.argmax(preds))
                predicted_brand = current_classes[predicted_idx]

                if predicted_brand.lower() == brand.lower():
                    brand_correct += 1
                    total_correct += 1

                total_tested += 1
            except Exception as e:
                print(f"  Error testing {img_path}: {e}")
                continue

        accuracy = (brand_correct / len(images) * 100) if images else 0
        status = "✓" if accuracy >= 80 else "⚠️" if accuracy >= 50 else "✗"
        print(f"  {status} {brand:15s}: {brand_correct}/{len(images)} correct ({accuracy:.0f}%)")

    overall = (total_correct / total_tested * 100) if total_tested > 0 else 0
    print(f"\n  Overall: {total_correct}/{total_tested} ({overall:.1f}%)")

    if overall < 60:
        print("\n🔴 MODEL PERFORMANCE IS POOR!")
        print("\nPossible causes:")
        print("  1. Model wasn't trained long enough")
        print("  2. Dataset has poor quality images")
        print("  3. Images are mislabeled in dataset folders")

        print("\n📋 RECOMMENDED ACTIONS:")
        print("\n  A. Manually verify dataset quality:")
        print("     • Open dataset/toyota/ (or any brand)")
        print("     • Check 10 random images")
        print("     • Confirm they're clear Toyota logo photos")

        print("\n  B. Retrain with more epochs:")
        print("     • Edit train.py: Change EPOCHS = 12 to EPOCHS = 20")
        print("     • Delete old model: rm emblemize_net.h5")
        print("     • Retrain: python train.py")
    else:
        print("\n✅ Model works well on training data!")
        print("\nIf predictions fail on NEW images, the issue is:")
        print("  • Test images are too different from training images")
        print("  • Test images are low quality, blurry, or poorly lit")
        print("  • Test images show full cars instead of just logos")

print("\n" + "=" * 70)