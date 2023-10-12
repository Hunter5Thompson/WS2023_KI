import tensorflow as tf
import numpy as np

# Mehr Daten generieren
x_train = np.random.randint(0, 100, (10000, 2))
y_train = np.sum(x_train, axis=1)

# Modell erstellen
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Adam-Optimizer mit angepasster Lernrate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Modell kompilieren
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Modell trainieren
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)

# Vorhersage
print(model.predict([[2, 3]]))  # Sollte sehr nahe an 5 sein
