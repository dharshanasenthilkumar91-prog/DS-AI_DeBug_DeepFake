import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import warnings

# REMOVE ALL WARNINGS
warnings.filterwarnings("ignore")

# LOAD MODEL WITHOUT COMPILING → Removes the Absl WARNING
model = tf.keras.models.load_model("deepfake_detector.h5", compile=False)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # REMOVE PROGRESS BAR
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    # CONVERT TO %
    percent = prediction * 100

    print("\n--- RESULT ---")
    print(f"Model confidence: {percent:.2f}%")

    if prediction > 0.7:
        print("Prediction: REAL (High Confidence)")
    elif prediction > 0.5:
        print("Prediction: REAL (Low Confidence)")
    else:
        print("Prediction: FAKE")

while True:
    path = input("\nEnter image path (or 'q' to quit): ").strip()
    if path.lower() == "q":
        break
    predict_image(path)

