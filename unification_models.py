import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import get_custom_objects

class Cast(tf.keras.layers.Layer):
    def __init__(self, target_dtype="float32", **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = tf.as_dtype(target_dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({"target_dtype": self.target_dtype.name})
        return config

get_custom_objects()['Cast'] = Cast

model_kp_path = "model_kp.h5"
model_class_path = "model_class.h5"

print("Загрузка модели для kp индекса...")
model_kp = load_model(model_kp_path, custom_objects={'Cast': Cast, 'mse': tf.keras.losses.MeanSquaredError()})
print("Загрузка модели для классификации снимка...")
model_class = load_model(model_class_path, custom_objects={'Cast': Cast})

for layer in model_kp.layers:
    layer.trainable = False
for layer in model_class.layers:
    layer.trainable = False

input_shape = (128, 128, 3)
input_layer = Input(shape=input_shape, name="input_image")

kp_output = model_kp(input_layer)
class_output = model_class(input_layer)

combined_model = Model(inputs=input_layer, outputs=[kp_output, class_output], name="combined_model")

combined_model.compile(
    optimizer="adam",
    loss=["mse", "categorical_crossentropy"],
    metrics=["mae", "accuracy"]
)

combined_model.summary()

combined_model.save("models/combined_model.h5")
