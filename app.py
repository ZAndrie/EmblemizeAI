# app.py - Maximum Speed Optimization
import gradio as gr
import numpy as np
from PIL import Image
import os
import sys
from functools import lru_cache
import time

MODEL_PATH = "emblemize_net.h5"
IMG_SIZE = (224, 224)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found!")
    print("Please train the model first by running: python train.py")
    sys.exit(1)

# Import TensorFlow after check
import tensorflow as tf
import cv2

print(f"TensorFlow version: {tf.__version__}")

# Configure TensorFlow for speed
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce logging

# Load model with proper handling
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Recompile with optimizations
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # Speed boost
    )
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load class names
try:
    with open("classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(class_names)} classes: {class_names}")
except Exception as e:
    print(f"WARNING: Could not load classes.txt: {e}")
    class_names = [f"Class_{i}" for i in range(model.output_shape[-1])]


# Pre-create TF function for faster inference
@tf.function(reduce_retracing=True)
def fast_predict(x):
    """Compiled prediction function for speed"""
    return model(x, training=False)


# Cache for processed images to avoid reprocessing
image_cache = {}


def preprocess_image_fast(pil_image):
    """Ultra-fast preprocessing"""
    # Create cache key
    img_hash = hash(pil_image.tobytes())

    if img_hash in image_cache:
        return image_cache[img_hash]

    # Fast resize using thumbnail
    img = pil_image.convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.NEAREST)  # Fastest resampling

    # Direct numpy conversion
    arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)  # Faster than division

    # Cache result
    if len(image_cache) < 50:  # Limit cache size
        image_cache[img_hash] = arr

    return arr


def predict_only(image: Image.Image):
    """Fast prediction without Grad-CAM"""
    if image is None:
        return "No image", "0%"

    try:
        start = time.time()

        # Preprocess
        arr = preprocess_image_fast(image)
        x = np.expand_dims(arr, axis=0)

        # Fast inference using TF function
        preds = fast_predict(x).numpy()[0]

        # Get result
        top_idx = int(np.argmax(preds))
        prob = float(preds[top_idx])
        label = class_names[top_idx]

        elapsed = time.time() - start
        print(f"Prediction took {elapsed:.3f}s")

        return f"{label}", f"{prob * 100:.2f}%"

    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Error: {str(e)}", "0%"


def predict_with_gradcam(image: Image.Image):
    """Slower prediction with visualization"""
    if image is None:
        return "No image", "0%", None

    try:
        arr = preprocess_image_fast(image)
        x = np.expand_dims(arr, axis=0)

        # Get prediction
        preds = fast_predict(x).numpy()[0]
        top_idx = int(np.argmax(preds))
        prob = float(preds[top_idx])
        label = class_names[top_idx]

        # Generate Grad-CAM
        heat = generate_gradcam(image, x)

        return f"{label}", f"{prob * 100:.2f}%", heat

    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}", "0%", None


def generate_gradcam(image, x):
    """Generate Grad-CAM (slow, optional)"""
    try:
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

        if not last_conv:
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x, training=False)
            top_pred = tf.argmax(predictions[0])
            top_class = predictions[:, top_pred]

        grads = tape.gradient(top_class, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap.numpy(), IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        orig = np.asarray(image.convert("RGB").resize(IMG_SIZE, Image.Resampling.NEAREST))
        result = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        return Image.fromarray(result)
    except:
        return None


# GRADIO UI - Optimized for speed
with gr.Blocks(title="EmblemizeAI") as demo:
    gr.Markdown("# 🚗 EmblemizeAI — Car Brand Detector")
    gr.Markdown("Upload a photo of a car emblem or front grill. Fast prediction mode enabled!")

    with gr.Row():
        img_in = gr.Image(type="pil", label="Upload or take photo")

    with gr.Row():
        with gr.Column():
            predict_fast_btn = gr.Button("⚡ Quick Detect (Fast)", variant="primary", size="lg")
        with gr.Column():
            predict_viz_btn = gr.Button("🔍 Detect + Visualization (Slow)", variant="secondary", size="lg")

    with gr.Row():
        with gr.Column():
            lbl = gr.Textbox(label="Predicted Brand", interactive=False)
            conf = gr.Textbox(label="Confidence Score", interactive=False)
        with gr.Column():
            gradcam_out = gr.Image(label="Attention Map (Grad-CAM)")

    gr.Markdown("**⚡ Quick Detect:** Instant prediction without visualization (~0.5-1s)")
    gr.Markdown("**🔍 Detect + Visualization:** Includes attention map but slower (~3-5s)")

    # Fast prediction (no Grad-CAM)
    predict_fast_btn.click(
        fn=lambda img: (*predict_only(img), None),
        inputs=[img_in],
        outputs=[lbl, conf, gradcam_out]
    )

    # Slow prediction (with Grad-CAM)
    predict_viz_btn.click(
        fn=predict_with_gradcam,
        inputs=[img_in],
        outputs=[lbl, conf, gradcam_out]
    )

    gr.Markdown(f"**Trained classes:** {', '.join(class_names)}")
    gr.Markdown("**Tips:** Use well-lit photos. Crop close to the emblem for best results.")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting EmblemizeAI server (Optimized)...")
    print("=" * 60 + "\n")

    # Warm up model with dummy prediction
    print("Warming up model...")
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    _ = fast_predict(dummy)
    print("✓ Model ready!\n")

    demo.queue(
        max_size=20,
        default_concurrency_limit=10
    )
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=True
    )