# Create your first MLP in Keras
import tensorflow as tf
import numpy as np
import sklearn.datasets as skdataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# obtiene las imagenes y su clasificacion
bunch_dataset = skdataset.load_files("train-shapes/")

# inicializa array de tensores
bunch_dataset["image_tensor"] = []

# realiza una decodificacion de la imagen en PNG a un tensor normalizado
with tf.Session() as sess:
    for image_png in bunch_dataset['data']:
        bunch_dataset["image_tensor"].append(sess.run(tf.image.decode_png(image_png, channels=1)) / 255.0)
        print('Loading image ', len(bunch_dataset["image_tensor"]))

# separa datos para entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(bunch_dataset['image_tensor'], bunch_dataset['target'],
                                                    test_size=0.25)

# crea el modelo a usar en Keras (capas secuenciales sin recurrencias)
model = tf.keras.Sequential()

# arquitectura ConvNet
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# input como vector de 784 valores
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

# Define las capas con la cantidad de neuronas de cada una y su funcion de activacion
# model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(32, activation='sigmoid'))

model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))

# probabilidad de cada clase
model.add(tf.keras.layers.Dense(5, activation='softmax'))

# Compila el modelo definiendo optimizador, funcion objetivo y metrica a evaluar
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrena el modelo
history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test), epochs=100)

# imprime en consola un resumen del entrenamiento
print('-----------------------------------------------------------')
print(model.summary())
print(history)
print('-----------------------------------------------------------')

# grafica el valor de precision durante el entrenamiento y la validacion de cada epoca
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test-shapes')
pyplot.legend()
pyplot.show()

# predice para 5 ejemplos dificiles
print('-----------------------------------------------------------')

bunch_datatest = skdataset.load_files("test-shapes/")
bunch_datatest["image_tensor"] = []

with tf.Session() as sess:
    for image_png in bunch_datatest['data']:
        bunch_datatest["image_tensor"].append(sess.run(tf.image.decode_png(image_png, channels=1)) / 255.0)

predictions = model.predict(np.array(bunch_datatest["image_tensor"]))
print(predictions)
print([(np.argmax(prediction), bunch_datatest["target"][index]) for index, prediction in enumerate(predictions)])
print('-----------------------------------------------------------')

# ---------------------- salidas de capa de convolucion y pooling -----------------------------

# extrae salida de cada capa
layer_outputs = [layer.output for layer in model.layers]
# genera un nuevo modelo para obtener las salidas
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# obtiene parametros para un ejemplo de validacion
activations = activation_model.predict(np.expand_dims(X_test[0], axis=0))

# realiza gráficas de la salida de cada capa (convolucion y pooling)
first_layer_activation = activations[0]
second_layer_activation = activations[1]
third_layer_activation = activations[2]
fourth_layer_activation = activations[3]

# imagen original
pyplot.matshow(np.array(X_test[0])[:, :, 0], cmap='gray')

# salida primera capa - conv2d 32 neuronas
pyplot.figure(figsize=(28, 28))
for index in range(32):
    pyplot.subplot(4, 8, index + 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(first_layer_activation[0, :, :, index], cmap=pyplot.cm.binary)
pyplot.show()

# salida segunda capa - pooling
pyplot.figure(figsize=(28, 28))
for index in range(32):
    pyplot.subplot(4, 8, index + 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(second_layer_activation[0, :, :, index], cmap=pyplot.cm.binary)
pyplot.show()

# salida tercera capa - conv2d 64 neuronas
pyplot.figure(figsize=(28, 28))
for index in range(64):
    pyplot.subplot(8, 8, index + 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(third_layer_activation[0, :, :, index], cmap=pyplot.cm.binary)
pyplot.show()

# salida cuarta capa - pooling
pyplot.figure(figsize=(28, 28))
for index in range(64):
    pyplot.subplot(8, 8, index + 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(fourth_layer_activation[0, :, :, index], cmap=pyplot.cm.binary)
pyplot.show()
