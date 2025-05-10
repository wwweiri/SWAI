#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from astropy.io import fits
from glymur import Jp2k

def convert_fits_to_image(fits_path, target_size=(128, 128)):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    data = np.array(data, dtype=np.float32)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    data = (data - min_val) / (max_val - min_val + 1e-8)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)

    image_resized = cv2.resize(data, target_size, interpolation=cv2.INTER_AREA)
    return image_resized

def load_jp2_image(filepath, target_size=(128, 128)):
    try:
        jp2 = Jp2k(filepath)
        image = jp2[:]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        min_val, max_val = np.min(image), np.max(image)
        image = (image - min_val) / (max_val - min_val + 1e-8)
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        if image_resized.ndim == 2:
            image_resized = np.expand_dims(image_resized, axis=-1)
        if image_resized.shape[-1] == 1:
            image_resized = np.repeat(image_resized, 3, axis=-1)
        return image_resized
    except:
        return None

def prepare_image_for_model(image):
    return np.expand_dims(image, axis=0)

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    file_path = sys.argv[1]
    target_size = (128, 128)

    if not os.path.exists(file_path):
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".fits":
        image = convert_fits_to_image(file_path, target_size)
    elif ext == ".jp2":
        image = load_jp2_image(file_path, target_size)
    else:
        sys.exit(1)

    if image is None:
        sys.exit(1)

    image_to_save = (image * 255).astype(np.uint8)
    cv2.imwrite("converted_image.png", cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
    prepared_image = prepare_image_for_model(image)

    model = tf.keras.models.load_model("models/final_model.h5")
    predictions = model.predict(prepared_image)
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)

if __name__ == "__main__":
    main()
