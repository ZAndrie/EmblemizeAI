# main.py - EmblemizeAI Mobile App with Camera & TFLite
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.clock import Clock
import numpy as np
from PIL import Image
import io
import os

# Set window size for testing on desktop
Window.size = (360, 640)

# Try to import TensorFlow Lite first (lighter for mobile)
USE_TFLITE = False
try:
    import tflite_runtime.interpreter as tflite

    USE_TFLITE = True
    print("✓ Using TensorFlow Lite Runtime")
except ImportError:
    try:
        import tensorflow as tf

        print("✓ Using TensorFlow")
    except ImportError:
        print("ERROR: No TensorFlow found!")

# Model configuration
TFLITE_MODEL_PATH = "emblemize_model.tflite"
H5_MODEL_PATH = "emblemize_net.h5"
IMG_SIZE = (224, 224)


class EmblemizeApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.interpreter = None
        self.class_names = []
        self.current_image = None
        self.camera_popup = None

    def build(self):
        self.title = "EmblemizeAI - Car Brand Detector"

        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Header
        header = Label(
            text='🚗 EmblemizeAI',
            size_hint=(1, 0.08),
            font_size='24sp',
            bold=True
        )
        layout.add_widget(header)

        # Image display area
        self.image_widget = KivyImage(
            size_hint=(1, 0.35),
            source='',
            allow_stretch=True
        )
        layout.add_widget(self.image_widget)

        # Result labels
        self.result_label = Label(
            text='No prediction yet',
            size_hint=(1, 0.12),
            font_size='20sp',
            bold=True,
            color=(0.2, 0.6, 1, 1)
        )
        layout.add_widget(self.result_label)

        self.confidence_label = Label(
            text='Confidence: --',
            size_hint=(1, 0.08),
            font_size='16sp',
            color=(0.5, 0.5, 0.5, 1)
        )
        layout.add_widget(self.confidence_label)

        # Buttons layout
        button_layout = BoxLayout(
            orientation='vertical',
            size_hint=(1, 0.32),
            spacing=8
        )

        # Camera button
        camera_btn = Button(
            text='📸 Take Photo',
            font_size='18sp',
            background_color=(1, 0.4, 0.2, 1),
            size_hint=(1, 0.33)
        )
        camera_btn.bind(on_press=self.open_camera)
        button_layout.add_widget(camera_btn)

        # Select image button
        select_btn = Button(
            text='🖼️ Select from Gallery',
            font_size='18sp',
            background_color=(0.2, 0.6, 1, 1),
            size_hint=(1, 0.33)
        )
        select_btn.bind(on_press=self.open_file_chooser)
        button_layout.add_widget(select_btn)

        # Predict button
        self.predict_btn = Button(
            text='🔍 Detect Brand',
            font_size='18sp',
            background_color=(0.2, 0.8, 0.2, 1),
            size_hint=(1, 0.33),
            disabled=True
        )
        self.predict_btn.bind(on_press=self.predict_brand)
        button_layout.add_widget(self.predict_btn)

        layout.add_widget(button_layout)

        # Model info at bottom
        self.info_label = Label(
            text='Loading model...',
            size_hint=(1, 0.05),
            font_size='12sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        layout.add_widget(self.info_label)

        # Load model
        Clock.schedule_once(lambda dt: self.load_model(), 0.5)

        return layout

    def load_model(self):
        """Load the trained model (TFLite or H5)"""
        try:
            # Try TFLite first (smaller, faster)
            if USE_TFLITE and os.path.exists(TFLITE_MODEL_PATH):
                self.interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("✓ TFLite model loaded")
                self.info_label.text = f"Model: TFLite ({os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024):.1f}MB)"

            # Fallback to H5 model
            elif os.path.exists(H5_MODEL_PATH):
                self.model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
                print("✓ H5 model loaded")
                self.info_label.text = f"Model: TensorFlow ({os.path.getsize(H5_MODEL_PATH) / (1024 * 1024):.1f}MB)"

            else:
                self.result_label.text = "⚠️ No model found!"
                self.result_label.color = (1, 0.3, 0.3, 1)
                self.info_label.text = "Place emblemize_model.tflite or .h5 in app folder"
                return

            # Load class names
            if os.path.exists("classes.txt"):
                with open("classes.txt", "r") as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                print(f"✓ Loaded {len(self.class_names)} classes: {self.class_names}")
            else:
                num_classes = (self.output_details[0]['shape'][-1] if self.interpreter
                               else self.model.output_shape[-1])
                self.class_names = [f"Class_{i}" for i in range(num_classes)]

            self.result_label.text = "Ready! Take photo or select image"
            self.result_label.color = (0.2, 0.8, 0.2, 1)

        except Exception as e:
            print(f"Error loading model: {e}")
            self.result_label.text = f"Error loading model"
            self.result_label.color = (1, 0.3, 0.3, 1)
            self.info_label.text = str(e)

    def open_camera(self, instance):
        """Open camera to take photo"""
        content = BoxLayout(orientation='vertical')

        # Camera widget
        camera = Camera(
            resolution=(640, 480),
            play=True,
            size_hint=(1, 0.85)
        )
        content.add_widget(camera)

        # Buttons
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)

        capture_btn = Button(
            text='📸 Capture',
            background_color=(0.2, 0.8, 0.2, 1)
        )
        cancel_btn = Button(
            text='Cancel',
            background_color=(0.8, 0.2, 0.2, 1)
        )

        btn_layout.add_widget(capture_btn)
        btn_layout.add_widget(cancel_btn)
        content.add_widget(btn_layout)

        # Create popup
        self.camera_popup = Popup(
            title='Take Photo of Car Emblem',
            content=content,
            size_hint=(0.95, 0.95)
        )

        def on_capture(instance):
            # Export camera image
            camera.export_to_png("temp_camera.png")
            Clock.schedule_once(lambda dt: self.load_camera_image(), 0.1)
            self.camera_popup.dismiss()

        def on_cancel(instance):
            camera.play = False
            self.camera_popup.dismiss()

        capture_btn.bind(on_press=on_capture)
        cancel_btn.bind(on_press=on_cancel)

        self.camera_popup.open()

    def load_camera_image(self):
        """Load image captured from camera"""
        if os.path.exists("temp_camera.png"):
            self.load_image("temp_camera.png")

    def open_file_chooser(self, instance):
        """Open file chooser to select image"""
        content = BoxLayout(orientation='vertical')

        # File chooser
        filechooser = FileChooserIconView(
            filters=['*.png', '*.jpg', '*.jpeg'],
            path=os.path.expanduser('~')
        )
        content.add_widget(filechooser)

        # Buttons
        btn_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)

        select_btn = Button(text='Select', background_color=(0.2, 0.6, 1, 1))
        cancel_btn = Button(text='Cancel', background_color=(0.8, 0.2, 0.2, 1))

        btn_layout.add_widget(select_btn)
        btn_layout.add_widget(cancel_btn)
        content.add_widget(btn_layout)

        # Create popup
        popup = Popup(
            title='Select Car Image',
            content=content,
            size_hint=(0.9, 0.9)
        )

        def on_select(instance):
            if filechooser.selection:
                self.load_image(filechooser.selection[0])
                popup.dismiss()

        def on_cancel(instance):
            popup.dismiss()

        select_btn.bind(on_press=on_select)
        cancel_btn.bind(on_press=on_cancel)

        popup.open()

    def load_image(self, path):
        """Load and display selected image"""
        try:
            # Load image
            self.current_image = Image.open(path).convert('RGB')

            # Display image
            self.image_widget.source = path
            self.image_widget.reload()

            # Enable predict button
            self.predict_btn.disabled = False
            self.result_label.text = "Image loaded! Tap 'Detect Brand'"
            self.result_label.color = (0.2, 0.6, 1, 1)
            self.confidence_label.text = "Confidence: --"

        except Exception as e:
            print(f"Error loading image: {e}")
            self.result_label.text = f"Error loading image"
            self.result_label.color = (1, 0.3, 0.3, 1)

    def preprocess_image(self, pil_image):
        """Preprocess image for model"""
        img = pil_image.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict_brand(self, instance):
        """Predict car brand from loaded image"""
        if self.current_image is None:
            return

        if self.interpreter is None and self.model is None:
            self.result_label.text = "Model not loaded!"
            return

        try:
            # Show loading
            self.result_label.text = "🔍 Analyzing..."
            self.result_label.color = (1, 0.7, 0, 1)

            # Preprocess
            x = self.preprocess_image(self.current_image)

            # Predict using TFLite or TensorFlow
            if self.interpreter:
                # TFLite inference
                self.interpreter.set_tensor(self.input_details[0]['index'], x)
                self.interpreter.invoke()
                preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                # TensorFlow inference
                preds = self.model.predict(x, verbose=0)[0]

            # Get top prediction
            top_idx = int(np.argmax(preds))
            prob = float(preds[top_idx])
            label = self.class_names[top_idx] if top_idx < len(self.class_names) else f"Class_{top_idx}"

            # Display results
            self.result_label.text = f"🚗 {label}"
            self.result_label.color = (0.2, 0.8, 0.2, 1)
            self.confidence_label.text = f"Confidence: {prob * 100:.2f}%"

            # Show top 3 predictions
            top3_indices = np.argsort(preds)[-3:][::-1]
            top3_text = "\n".join([
                f"{i + 1}. {self.class_names[idx]}: {preds[idx] * 100:.1f}%"
                for i, idx in enumerate(top3_indices)
            ])
            print(f"Top 3 predictions:\n{top3_text}")

        except Exception as e:
            print(f"Prediction error: {e}")
            self.result_label.text = "Prediction failed!"
            self.result_label.color = (1, 0.3, 0.3, 1)
            self.confidence_label.text = f"Error: {str(e)}"


if __name__ == '__main__':
    EmblemizeApp().run()