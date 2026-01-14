# convert_to_tflite.py - Convert H5 model to TensorFlow Lite
import tensorflow as tf
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("TensorFlow Lite Converter")
print("=" * 60)

MODEL_PATH = "emblemize_net.h5"
TFLITE_PATH = "emblemize_model.tflite"
IMG_SIZE = (224, 224)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found!")
    print("Please train the model first.")
    exit(1)

print(f"\n1. Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"   ✓ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

print("\n2. Converting to TensorFlow Lite...")
try:
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization options (reduces size significantly)
    print("   Applying optimizations...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Use float16 quantization for even smaller size
    # converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    print(f"   ✓ Conversion successful")
    
except Exception as e:
    print(f"   ✗ Conversion failed: {e}")
    exit(1)

print("\n3. Saving TFLite model...")
try:
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    h5_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    reduction = ((h5_size - tflite_size) / h5_size) * 100
    
    print(f"   ✓ Saved to {TFLITE_PATH}")
    print(f"\n   Original H5 size:  {h5_size:.2f} MB")
    print(f"   TFLite size:       {tflite_size:.2f} MB")
    print(f"   Size reduction:    {reduction:.1f}%")
    
except Exception as e:
    print(f"   ✗ Error saving: {e}")
    exit(1)

print("\n4. Testing TFLite model...")
try:
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   ✓ TFLite model loaded successfully")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Test with dummy data
    dummy_input = np.random.random((1, IMG_SIZE[0], IMG_SIZE[1], 3)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   ✓ Test inference successful")
    print(f"   Output shape: {output.shape}")
    
except Exception as e:
    print(f"   ✗ Testing failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Conversion Complete! ✓")
print("=" * 60)
print(f"\nFiles created:")
print(f"  • {TFLITE_PATH} ({tflite_size:.2f} MB)")
print(f"\nNext steps:")
print(f"  1. Copy {TFLITE_PATH} to your mobile app folder")
print(f"  2. Use the updated main.py (with TFLite support)")
print(f"  3. Update buildozer.spec requirements to use tflite-runtime")
print("=" * 60)
