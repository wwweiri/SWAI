import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.utils import get_custom_objects

class Cast(tf.keras.layers.Layer):
    def __init__(self, target_dtype="float32", **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "Cast_" + str(np.random.randint(10000, 99999))
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = tf.as_dtype(target_dtype)
    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)
    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype.name})
        return config

get_custom_objects()['Cast'] = Cast

model_kp_path = "models/model_kp.h5"
model_class_path = "models/model_class.h5"

def combine_image_models(model_kp_path, model_class_path, input_shape=(128, 128, 3)):
    model_kp = load_model(model_kp_path, custom_objects={'Cast': Cast, 'mse': tf.keras.losses.MeanSquaredError()})
    model_class = load_model(model_class_path, custom_objects={'Cast': Cast})

    input_layer = Input(shape=input_shape, name="input_image")
    kp_output = model_kp(input_layer)
    class_output = model_class(input_layer)

    combined_model = Model(inputs=input_layer, outputs=[kp_output, class_output], name="combined_image_model")
    return combined_model

def get_additional_parameters():
    """
    В этой функции возвращаем массив дополнительных параметров:
    - солнечный ветер (скорость) [например, 400 км/с]
    - магнитное поле Bz [например, -5 нТ]
    - Dst индекс [например, -50]
    Эти значения должны быть нормализованы по мере необходимости.
    """
    return np.array([400.0, -5.0, -50.0])

def build_feature_vector(image):
    """
    Из изображения (через объединенную модель) получаем:
      - kp индекс (одно число)
      - классификационные вероятности (например, 5 чисел)
    Затем получаем дополнительные параметры (3 числа)
    Итоговый вектор признаков имеет размер 1 + 5 + 3 = 9
    """
    combined_model = combine_image_models(model_kp_path, model_class_path)
    outputs = combined_model.predict(image)
    kp_pred = outputs[0][0][0]
    class_probs = outputs[1][0]

    additional_params = get_additional_parameters()
    feature_vector = np.concatenate([[kp_pred], class_probs, additional_params])
    return feature_vector

def build_impact_model(input_dim=9):
    input_features = Input(shape=(input_dim,), name="input_features")
    x = Dense(32, activation="relu")(input_features)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    output = Dense(3, activation="softmax", name="impact_class")(x)
    model = Model(inputs=input_features, outputs=output, name="impact_model")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    test_image_path = "D:\\sw\\AIA_304\\2015-01-01T03-00-00Z.jp2"
    import cv2
    def load_and_preprocess_jp2(image_path, target_size=(128,128)):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        if len(img.shape)==2:
            img = cv2.merge([img]*3)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255.0
        return np.expand_dims(img, axis=0)

    image = load_and_preprocess_jp2(test_image_path, target_size=(128,128))

    feature_vector = build_feature_vector(image)
    print("Вектор признаков:", feature_vector)

    impact_model = build_impact_model(input_dim=feature_vector.shape[0])
    impact_model.summary()

    risk_prediction = impact_model.predict(np.expand_dims(feature_vector, axis=0))
    risk_class = np.argmax(risk_prediction)
    risk_mapping = {0: "Низкий риск", 1: "Средний риск", 2: "Высокий риск"}
    print(f"Предсказанный риск влияния на инфраструктуру: {risk_mapping.get(risk_class, 'Неизвестно')}")

if __name__ == "__main__":
    main()
