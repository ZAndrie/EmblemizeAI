import tensorflow as tf

print("Loading H5 model...")
model = tf.keras.models.load_model("emblemize_net.h5")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional but recommended)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("emblemize_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ emblemize_model.tflite created successfully!")
print("File size:", len(tflite_model) / 1024 / 1024, "MB")
