import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("dog_cat_model.h5")
classes = ["cat", "dog"]
THRESHOLD = 0.60

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    confidence = np.max(preds)

    if confidence < THRESHOLD:
        return "Neither Cat nor Dog – Please upload a valid image.", confidence

    label = classes[np.argmax(preds)]
    return f"This is a {label.capitalize()}", confidence
