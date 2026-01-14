# deep_diagnostic.py - Find the REAL problem with your model
import os
import sys
import numpy as np
from PIL import Image
import random

print("=" * 70)
print("DEEP DIAGNOSTIC - Finding the Real Problem")
print("=" * 70)

# Check TensorFlow
try:
    import tensorflow as tf

    print(f"\n✓ TensorFlow {tf.__version__}")
except ImportError:
    print("\n✗ TensorFlow not installed!")
    sys.exit(1)

# Load model
print("\n1. Loading Model...")
print("-" * 70)

if not os.path.exists("emblemize_net.h5"):
    print("✗ emblemize_net.h5 not found!")
    sys.exit(1)

try:
    model = tf.keras.models.load_model("emblemize_net.h5", compile=False)
    print(f"✓ Model loaded")
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")
    num_classes = model.output_shape[-1]
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Load classes
print("\n2. Loading Classes...")
print("-" * 70)

if not os.path.exists("classes.txt"):
    print("✗ classes.txt not found!")
    sys.exit(1)

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ Loaded {len(class_names)} classes")
print(f"  Classes: {class_names}")

if len(class_names) != num_classes:
    print(f"\n❌ CRITICAL PROBLEM FOUND!")
    print(f"  classes.txt has {len(class_names)} classes")
    print(f"  Model expects {num_classes} classes")
    print(f"\n  FIX: Delete emblemize_net.h5 and retrain")
    print(f"       rm emblemize_net.h5")
    print(f"       python train_kaggle.py")
    sys.exit(1)

# Check dataset
print("\n3. Analyzing Dataset...")
print("-" * 70)

if not os.path.exists("dataset"):
    print("✗ dataset/ folder not found!")
    sys.exit(1)

brand_folders = [d for d in os.listdir("dataset")
                 if os.path.isdir(os.path.join("dataset", d))]

print(f"✓ Found {len(brand_folders)} brand folders")

# Check each brand
brand_images = {}
for brand in brand_folders:
    path = os.path.join("dataset", brand)
    images = [f for f in os.listdir(path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    brand_images[brand] = images
    print(f"  {brand:20s}: {len(images):4d} images")

# Check brand names match
dataset_brands = set(brand_folders)
model_brands = set(class_names)

if dataset_brands != model_brands:
    print(f"\n❌ CRITICAL PROBLEM FOUND!")
    print(f"  Dataset brands: {sorted(dataset_brands)}")
    print(f"  Model classes:  {sorted(model_brands)}")
    print(f"\n  Brands in dataset but NOT in model: {dataset_brands - model_brands}")
    print(f"  Brands in model but NOT in dataset: {model_brands - dataset_brands}")
    print(f"\n  FIX: Your model was trained on different brands!")
    print(f"       Delete emblemize_net.h5 and retrain:")
    print(f"       rm emblemize_net.h5")
    print(f"       python train_kaggle.py")
    sys.exit(1)

# Deep test: Check actual image content
print("\n4. Deep Testing: Checking Image Quality...")
print("-" * 70)


def preprocess_image(img_path):
    """Preprocess exactly like the app"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# Test random images from each brand
print("\nTesting random images from each brand...\n")

total_tested = 0
total_correct = 0
brand_accuracy = {}

for brand in sorted(brand_folders):
    if brand not in class_names:
        continue

    images = brand_images[brand]
    if len(images) == 0:
        continue

    # Test 5 random images from this brand
    test_count = min(5, len(images))
    test_images = random.sample(images, test_count)

    correct = 0
    confidences = []

    for img_name in test_images:
        img_path = os.path.join("dataset", brand, img_name)
        try:
            x = preprocess_image(img_path)
            preds = model.predict(x, verbose=0)[0]

            predicted_idx = int(np.argmax(preds))
            predicted_brand = class_names[predicted_idx]
            confidence = float(preds[predicted_idx])

            confidences.append(confidence)

            if predicted_brand.lower() == brand.lower():
                correct += 1

        except Exception as e:
            print(f"  ✗ Error with {img_path}: {e}")
            continue

    accuracy = (correct / test_count) * 100
    avg_conf = np.mean(confidences) * 100 if confidences else 0

    brand_accuracy[brand] = accuracy
    total_tested += test_count
    total_correct += correct

    status = "✓" if accuracy >= 80 else "⚠" if accuracy >= 50 else "✗"
    print(f"  {status} {brand:20s}: {correct}/{test_count} correct ({accuracy:.0f}%) | Avg confidence: {avg_conf:.1f}%")

overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0

print(f"\n{'=' * 70}")
print(f"OVERALL ACCURACY: {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
print(f"{'=' * 70}")

# Diagnose the problem
print("\n5. DIAGNOSIS")
print("-" * 70)

if overall_accuracy >= 80:
    print("\n✅ Model is working WELL!")
    print("   If you're getting wrong predictions in the app, the problem")
    print("   might be with the images you're testing with.")
    print("\n   Try:")
    print("   - Using images similar to your training data")
    print("   - Close-up photos of car logos/emblems")
    print("   - Well-lit, clear images")

elif overall_accuracy >= 50:
    print("\n⚠ Model is MEDIOCRE - needs improvement")
    print("\n   Possible causes:")
    print("   1. Training didn't converge properly")
    print("   2. Images have too much variety/noise")
    print("   3. Need more epochs")

    print("\n   TRY THIS:")
    print("   1. Check a few images manually - are they clear car logos?")
    print("   2. Retrain with MORE epochs:")
    print("      python train_kaggle.py")
    print("   3. Make sure training shows INCREASING accuracy")

else:
    print("\n❌ Model has SERIOUS PROBLEMS!")
    print("\n   This suggests:")
    print("   1. Model wasn't trained properly")
    print("   2. Images might not be what you think they are")
    print("   3. Preprocessing mismatch")

    print("\n   IMMEDIATE ACTIONS:")
    print("\n   A. Check your images RIGHT NOW:")
    print("      - Open dataset/toyota/ (or any brand)")
    print("      - Look at 5-10 random images")
    print("      - Are they CLEAR PHOTOS of Toyota logos/emblems?")
    print("      - Or are they something else?")

    print("\n   B. Check training output:")
    print("      When you ran train_kaggle.py, did you see:")
    print("      'Training Accuracy: 85%' or similar?")
    print("      If not, training failed!")

# Show worst performing brands
print("\n6. Worst Performing Brands:")
print("-" * 70)

sorted_brands = sorted(brand_accuracy.items(), key=lambda x: x[1])
print("\nThese brands have lowest accuracy:")
for brand, acc in sorted_brands[:5]:
    print(f"  {brand:20s}: {acc:.0f}%")

print("\n   ACTION: Manually check images in these folders!")
print("   They might have wrong/corrupted images")

# Sample prediction visualization
print("\n7. Sample Prediction Details:")
print("-" * 70)

# Pick one brand and show detailed predictions
test_brand = brand_folders[0]
test_images = brand_images[test_brand][:3]

print(f"\nDetailed look at {test_brand} predictions:\n")

for img_name in test_images:
    img_path = os.path.join("dataset", test_brand, img_name)
    try:
        x = preprocess_image(img_path)
        preds = model.predict(x, verbose=0)[0]

        # Get top 5 predictions
        top5_idx = np.argsort(preds)[-5:][::-1]

        print(f"Image: {img_name}")
        print(f"  True brand: {test_brand}")
        print(f"  Top 5 predictions:")
        for i, idx in enumerate(top5_idx):
            print(f"    {i + 1}. {class_names[idx]:15s}: {preds[idx] * 100:5.1f}%")
        print()

    except Exception as e:
        print(f"  Error: {e}\n")

# Final recommendations
print("\n" + "=" * 70)
print("FINAL RECOMMENDATIONS")
print("=" * 70)

if overall_accuracy < 60:
    print("\n🔴 YOUR MODEL IS NOT WORKING")
    print("\n   Step 1: VERIFY YOUR IMAGES")
    print("   - Go to dataset/ folder")
    print("   - Open 10 random images from different brands")
    print("   - Confirm they are clear car logo photos")
    print("   - If they look wrong, you have the wrong dataset!")

    print("\n   Step 2: DELETE OLD MODEL AND RETRAIN")
    print("   Run these commands:")
    print("   >>> rm emblemize_net.h5")
    print("   >>> rm best_model.h5")
    print("   >>> python train_kaggle.py")

    print("\n   Step 3: WATCH THE TRAINING")
    print("   You should see accuracy going UP each epoch:")
    print("   Epoch 5: Training: 65%, Validation: 62%")
    print("   Epoch 10: Training: 82%, Validation: 78%")
    print("   Epoch 15: Training: 90%, Validation: 85%")

    print("\n   If training accuracy stays LOW, your images are wrong!")

else:
    print("\n🟡 Model works on training data but maybe not on new images?")
    print("\n   Try testing with:")
    print("   - Images FROM your training dataset first")
    print("   - Similar quality/style to your training images")
    print("   - Close-up car logo photos (not full car photos)")

print("\n" + "=" * 70)