model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=[input_shape]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(output_shape) # No activation on the final layer for regression
])