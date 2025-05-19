import os
import random
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import get_custom_objects

class Cast(tf.keras.layers.Layer):
    def __init__(self, target_dtype="float32", **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "Cast_" + str(random.randint(10000, 99999))
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = tf.as_dtype(target_dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype.name})
        return config

get_custom_objects()['Cast'] = Cast

def find_random_jp2_file(root_dir):
    jp2_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jp2"):
                jp2_files.append(os.path.join(subdir, file))
    if not jp2_files:
        raise FileNotFoundError("Не найдено ни одного .jp2 файла.")
    return random.choice(jp2_files)

def load_and_preprocess_jp2(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    if len(img.shape) == 2:
        img = cv2.merge([img] * 3)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def get_true_kp_from_csv(csv_path, image_path):
    import pandas as pd
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    rel_path = os.path.relpath(image_path, start=r"D:\sw")
    csv_style_path = os.path.join("/mnt/remote_fits", rel_path).replace("\\", "/")
    print(f"Преобразованный путь для поиска в CSV: {csv_style_path}")
    row = df[df['image_path'] == csv_style_path]
    if row.empty:
        print("⚠️ Не найдено соответствие в CSV для:", csv_style_path)
        return None
    return row.iloc[0]['Kp']

def combine_models(model_kp_path, model_class_path, input_shape=(128, 128, 3)):
    print("Загрузка модели для kp индекса...")
    model_kp = load_model(model_kp_path, custom_objects={'Cast': Cast, 'mse': tf.keras.losses.MeanSquaredError()})
    print("Загрузка модели для классификации снимка...")
    model_class = load_model(model_class_path, custom_objects={'Cast': Cast})

    for layer in model_kp.layers:
        layer.trainable = False
    for layer in model_class.layers:
        layer.trainable = False

    input_layer = Input(shape=input_shape, name="input_image")
    kp_output = model_kp(input_layer)
    class_output = model_class(input_layer)
    combined_model = Model(inputs=input_layer, outputs=[kp_output, class_output], name="combined_model")
    return combined_model

def main():
    model_kp_path = "models/model_kp.h5"
    model_class_path = "models/model_class.h5"
    csv_path = "merged_images_kp.csv"
    root_dir = r"D:\sw"

    class_mapping = {
        0: "AIA_171",
        1: "AIA_193",
        2: "AIA_304",
        3: "HMI_Magnetogram",
        4: "SOHO_LASCO_C2"
    }

    print("Объединяем модели в одну...")
    combined_model = combine_models(model_kp_path, model_class_path, input_shape=(128, 128, 3))

    combined_model.save("combined_model.keras")
    print("Объединенная модель сохранена в формате .keras")

    print("Поиск случайного .jp2 файла...")
    jp2_path = find_random_jp2_file(root_dir)
    filename = os.path.basename(jp2_path)
    print(f"Найден файл: {filename}")
    print(f"Полный путь: {jp2_path}")

    image = load_and_preprocess_jp2(jp2_path, target_size=(128, 128))
    outputs = combined_model.predict(image)
    kp_pred = outputs[0][0][0]
    class_probs = outputs[1][0]
    class_idx = int(np.argmax(class_probs))
    class_name = class_mapping.get(class_idx, f"Класс {class_idx}")

    print(f"🔮 Предсказанный kp индекс для файла {filename}: {kp_pred:.2f}")
    print(f"🔮 Предсказанный класс снимка: {class_name}")

    true_kp = get_true_kp_from_csv(csv_path, jp2_path)
    if true_kp is not None:
        print(f"✅ Реальный kp индекс (из CSV): {true_kp}")
        print(f"📊 Разница kp: {abs(kp_pred - true_kp):.2f}")

if __name__ == "__main__":
    main()
