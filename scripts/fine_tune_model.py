import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(path, target_size=(128, 128)):
    def _load_image(path_str):
        try:
            path_decoded = path_str.decode('utf-8')
        except AttributeError:
            path_decoded = path_str.numpy().decode('utf-8')

        if not os.path.exists(path_decoded):
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

        img = cv2.imread(path_decoded)
        if img is None:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img

    image = tf.py_function(func=_load_image, inp=[path], Tout=tf.float32)
    image.set_shape([target_size[0], target_size[1], 3])
    return image

def create_dataset(csv_path, batch_size=32, target_size=(128, 128), shuffle=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].values
    kp_values = df['Kp'].values.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, kp_values))

    def _process(path, kp):
        image = load_and_preprocess_image(path, target_size)
        return image, kp

    dataset = dataset.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def modify_model_for_regression(model):
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model

    if not model.inputs:
        inp = Input(shape=(128, 128, 3))
    else:
        inp = model.inputs[0]
    x = inp
    for layer in model.layers[:-1]:
        x = layer(x)
    new_output = Dense(1, activation='linear', name='kp_regression')(x)
    new_model = Model(inputs=inp, outputs=new_output)
    return new_model

def main():
    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, "..")

    csv_path = os.path.join(project_root, "data", "merged_images_kp.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_csv = os.path.join(project_root, "data", "train_merged_images_kp.csv")
    val_csv = os.path.join(project_root, "data", "val_merged_images_kp.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    batch_size = 32
    target_size = (128, 128)
    train_ds = create_dataset(train_csv, batch_size=batch_size, target_size=target_size, shuffle=True)
    val_ds = create_dataset(val_csv, batch_size=batch_size, target_size=target_size, shuffle=False)

    model_path = os.path.join(project_root, "final_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    base_model = tf.keras.models.load_model(model_path)

    new_model = modify_model_for_regression(base_model)
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='mse',
                      metrics=['mae'])

    epochs = 10
    history = new_model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    fine_tuned_path = os.path.join(project_root, "fine_tuned_model.h5")
    new_model.save(fine_tuned_path)

if __name__ == "__main__":
    main()
