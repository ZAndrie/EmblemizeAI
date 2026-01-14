# test_model.py - Simple test script
import os
import sys

print("=" * 60)
print("EmblemizeAI - Model Test Script")
print("=" * 60)

# Check files
print("\n1. Checking required files...")
required_files = {
    "emblemize_net.h5": "Trained model",
    "classes.txt": "Class names",
    "app.py": "Application script",
    "train.py": "Training script"
}

all_exist = True
for file, desc in required_files.items():
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✓ {file} ({desc}) - {size:,} bytes")
    else:
        print(f"  ✗ {file} ({desc}) - NOT FOUND")
        all_exist = False

# Check dataset
print("\n2. Checking dataset folder...")
if os.path.exists("dataset"):
    subdirs = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]
    if subdirs:
        print(f"  ✓ Found {len(subdirs)} class folders:")
        for d in subdirs:
            img_count = len([f for f in os.listdir(os.path.join("dataset", d))
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    - {d}: {img_count} images")
    else:
        print("  ✗ No class folders found in dataset/")
else:
    print("  ✗ dataset/ folder not found")

# Test model loading
print("\n3. Testing model loading...")
if os.path.exists("emblemize_net.h5"):
    try:
        import tensorflow as tf

        print(f"  TensorFlow version: {tf.__version__}")

        print("  Loading model...")
        try:
            model = tf.keras.models.load_model("emblemize_net.h5")
            print("  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ⚠ Standard loading failed: {e}")
            print("  Trying without compilation...")
            model = tf.keras.models.load_model("emblemize_net.h5", compile=False)
            print("  ✓ Model loaded (without compilation)")

        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
        print(f"  Number of classes: {model.output_shape[-1]}")

        # Test classes.txt
        if os.path.exists("classes.txt"):
            with open("classes.txt", "r") as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"  Classes: {classes}")

            if len(classes) == model.output_shape[-1]:
                print("  ✓ Number of classes matches model output")
            else:
                print(f"  ✗ Mismatch: classes.txt has {len(classes)} but model expects {model.output_shape[-1]}")

    except ImportError:
        print("  ✗ TensorFlow not installed")
        print("    Install with: pip install tensorflow")
    except Exception as e:
        print(f"  ✗ Error: {e}")
else:
    print("  ⚠ Model file not found - run train.py first")

# Final recommendations
print("\n" + "=" * 60)
print("Recommendations:")
print("=" * 60)

if not os.path.exists("emblemize_net.h5"):
    print("❌ TRAIN MODEL FIRST:")
    print("   python train.py")
elif not all_exist:
    print("⚠️  MISSING FILES - Check above")
else:
    print("✓ ALL CHECKS PASSED!")
    print("\n🚀 Ready to run:")
    print("   python app.py")
    print("\nThen open: http://localhost:7860")

print("=" * 60)