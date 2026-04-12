import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model

model = keras.models.load_model('zadatak_1_model.keras')

print("===================================")
image = Image.open('test.png').convert('L')
image = image.resize((28, 28))
image_array = np.array(image)

image_array = image_array.astype("float32") / 255
image_array = np.expand_dims(image_array, axis=0)
image_array = np.expand_dims(image_array, axis=-1)

prediction = np.argmax(model.predict(image_array))

print(f"Predicted: {prediction}")
print("===================================")