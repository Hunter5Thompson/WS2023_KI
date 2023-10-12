import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Daten laden
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Daten vorbereiten
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Modell erstellen
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Modell kompilieren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Modell trainieren
model.fit(x_train, y_train, epochs=5)

# 6. Evaluation
model.evaluate(x_test, y_test)

# Funktion zum Testen eines einzelnen Bildes
def test_single_image(img):
    img = np.expand_dims(img, axis=0)  # Dimension für den Batch hinzufügen
    prediction = model.predict(img)
    return np.argmax(prediction)  # Gibt die Ziffer mit der höchsten Wahrscheinlichkeit zurück

# Ein Beispielbild aus dem Testdatensatz nehmen
example_img = x_test[0]

# Das Beispielbild anzeigen
plt.imshow(example_img, cmap='gray')
plt.show()

# Vorhersage machen
result = test_single_image(example_img)
print(f"Das Modell sagt, dass diese Zahl eine {result} ist.")