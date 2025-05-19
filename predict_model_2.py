import os
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable, custom_object_scope

@register_keras_serializable()
class Cast(tf.keras.layers.Layer):
    def __init__(self, target_dtype="float32", **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = tf.as_dtype(target_dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype.name})
        return config

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
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    # приводим путь к относительному варианту, как в CSV
    rel_path = os.path.relpath(image_path, start=r"D:\sw")
    csv_style_path = os.path.join("/mnt/remote_fits", rel_path).replace("\\", "/")
    print(f"Преобразованный путь для поиска в CSV: {csv_style_path}")
    row = df[df['image_path'] == csv_style_path]
    if row.empty:
        print("⚠️ Не найдено соответствие в CSV для:", csv_style_path)
        return None
    return row.iloc[0]['Kp']

def main():
    parser = argparse.ArgumentParser(description="Предсказание Kp-индекса по .jp2-снимку")
    parser.add_argument("--image_path", "-i", required=True,
                        help="Путь к .jp2 файлу для предсказания")
    parser.add_argument("--model_path", "-m", required=True,
                        help="Путь к сохранённой модели (.h5)")
    parser.add_argument("--csv_path", "-c", required=True,
                        help="Путь к CSV с истинными значениями Kp")
    args = parser.parse_args()

    jp2_path = args.image_path
    if not os.path.exists(jp2_path):
        raise FileNotFoundError(f".jp2 файл не найден: {jp2_path}")
    filename = os.path.basename(jp2_path)
    print(f"Используется файл: {filename}")
    print(f"Полный путь: {jp2_path}")

    image = load_and_preprocess_jp2(jp2_path)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Модель не найдена: {args.model_path}")
    print("Загрузка модели...")

    custom_objs = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'Cast': Cast
    }
    with custom_object_scope(custom_objs):
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objs)
    print("Модель загружена. Выполняется предсказание...")

    prediction = model.predict(image)
    kp_pred = float(prediction[0][0])
    print(f"🔮 Предсказанный kp индекс для файла {filename}: {kp_pred:.2f}")

    true_kp = get_true_kp_from_csv(args.csv_path, jp2_path)
    if true_kp is not None:
        print(f"✅ Реальный kp индекс (из CSV): {true_kp}")
        print(f"📊 Разница: {abs(kp_pred - true_kp):.2f}")

if __name__ == "__main__":
    main()
