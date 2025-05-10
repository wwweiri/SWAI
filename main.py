import os
import math
import numpy as np
import tensorflow as tf
from glymur import Jp2k
import cv2
from sklearn.model_selection import train_test_split

# Включаем XLA для оптимизации графа
tf.config.optimizer.set_jit(True)

# Включаем Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def load_jp2_image(filepath, target_size=(128, 128)):
    """
    Считывает JP2-файл, нормализует и изменяет его размер до target_size.
    Если изображение одноканальное, добавляет ось каналов, а затем дублирует до 3-х каналов.
    Возвращает массив формы (target_size[0], target_size[1], 3) с dtype float32, значения в [0,1].
    """
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
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None

def get_filepaths_and_labels(base_dir):
    """
    Ожидается, что base_dir содержит подпапки, каждая из которых соответствует классу.
    Возвращает:
      - filepaths: список путей к файлам .jp2,
      - labels: список числовых меток для каждого файла,
      - class_names: список имен классов (названия подпапок).
    """
    filepaths = []
    labels = []
    class_names = sorted([name for name in os.listdir(base_dir)
                          if os.path.isdir(os.path.join(base_dir, name))])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".jp2"):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(label_map[class_name])
    return filepaths, labels, class_names

def data_generator(file_list, label_list, target_size=(128, 128)):
    """
    Генератор, который последовательно загружает изображение и возвращает его вместе с меткой.
    Выводит в консоль информацию о каждом 100-м файле.
    """
    for idx, (fp, label) in enumerate(zip(file_list, label_list)):
        if (idx + 1) % 100 == 0:
            print(f"[Generator] Обработка файла {idx + 1}/{len(file_list)}: {fp}")
        image = load_jp2_image(fp, target_size)
        if image is not None:
            yield image, label

if __name__ == "__main__":
    base_dir = "/mnt/remote_fits"  # Измените на актуальный путь
    target_size = (128, 128)
    batch_size = 64
    epochs = 1

    # Сбор файлов и меток
    filepaths, labels, class_names = get_filepaths_and_labels(base_dir)
    print(f"Найдено {len(filepaths)} изображений в {len(class_names)} классах: {class_names}")

    filepaths = np.array(filepaths)
    labels = np.array(labels)
    train_files, val_files, train_labels, val_labels = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Создаем датасеты с кэшированием для ускорения загрузки
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_files, train_labels, target_size),
        output_types=(tf.float32, tf.int32),
        output_shapes=((target_size[0], target_size[1], 3), ())
    ).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_files, val_labels, target_size),
        output_types=(tf.float32, tf.int32),
        output_shapes=((target_size[0], target_size[1], 3), ())
    ).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    import math
    steps_per_epoch = math.ceil(len(train_files) / batch_size)
    validation_steps = math.ceil(len(val_files) / batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(target_size[0], target_size[1], 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps
    )

    model.save("final_model.h5")
