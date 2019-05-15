# Create your first MLP in Keras
import tensorflow as tf
import numpy as np
import random
import sklearn.datasets as skdataset
from collections import Counter

# define cantidad de datos para entranamiento, validacion y testeo
TOTAL_DATA = 1000
TRAIN_DATA = 500
VALIDATE_DATA = 300
TEST_DATA = 200

# obtiene las imagenes y su clasificacion
bunch_dataset = skdataset.load_files("shapes/Test/")
labeled_dataset = []

# reacomoda la data para un mejor manejo dentro del codigo
for index in range(len(bunch_dataset['data'])):
    labeled_dataset.append({
        "image_png": bunch_dataset['data'][index],
        "file": bunch_dataset['filenames'][index],
        "image_tensor": None,
        "target": bunch_dataset['target'][index]
    })

# desordena los datos para que no queden las clases juntas
random.shuffle(labeled_dataset)

# realiza una decodificacion de la imagen en PNG a un tensor normalizado (solo de la data que se va a usar TOTAL_DATA)
with tf.Session() as sess:
    for data in labeled_dataset[:TOTAL_DATA]:
        data["image_tensor"] = sess.run(tf.image.decode_png(data['image_png'], channels=1)) / 255.0
        print('Load image: ', data['file'])

# crea el modelo a usar en Keras (capaz secuenciales sin recurrencias)
model = tf.keras.Sequential()

# input como vector de 784 valores
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

# Define las capas con la cantidad de neuronas de cada una y su funcion de activacion
model.add(tf.keras.layers.Dense(128, activation='relu'))

# descomentar para imagenes "imperfectas"
# model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(5, activation='softmax'))

# Compila el modelo definiendo optimizador, funcion objetivo y metrica a evaluar
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrena el modelo
history = model.fit(np.array([data['image_tensor'] for data in labeled_dataset[0:TRAIN_DATA]]),
                    np.array([data['target'] for data in labeled_dataset[0:TRAIN_DATA]]), epochs=200)

print(model.summary())
print(history.history)
print('-----------------------------------------------------------')

# Valida el modelo
print(
    model.evaluate(np.array([data['image_tensor'] for data in labeled_dataset[TRAIN_DATA:TRAIN_DATA + VALIDATE_DATA]]),
                   np.array([data['target'] for data in labeled_dataset[TRAIN_DATA:TRAIN_DATA + VALIDATE_DATA]])))
print('-----------------------------------------------------------')

# predice los valores de testeo
predict = model.predict(
    np.array([data['image_tensor'] for data in labeled_dataset[TRAIN_DATA + VALIDATE_DATA:TOTAL_DATA]]))

for index in range(len(predict)):
    print(bunch_dataset['target_names'][labeled_dataset[TRAIN_DATA + VALIDATE_DATA + index]['target']] + " - " +
          bunch_dataset['target_names'][np.argmax(predict[index])])

print(Counter([bunch_dataset['target_names'][np.argmax(pre)] for pre in predict]))