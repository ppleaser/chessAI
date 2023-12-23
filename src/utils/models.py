from tensorflow.keras import layers, Model, Input

def average_models(models):
    """
    Averages the weights of a list of models and returns a new model with the averaged weights.

    Parameters:
    models (list): A list of models to be averaged.

    Returns:
    averaged_model: A new model with the averaged weights.
    """
    averaged_model = create_model()

    weights = [model.get_weights() for model in models]
    averaged_weights = [sum(w) / len(w) for w in zip(*weights)]

    averaged_model.set_weights(averaged_weights)

    return averaged_model


def create_model():
    """
    Creates a convolutional neural network model for the chess AI.

    Returns:
        model (tensorflow.keras.Model): The compiled model.
    """
    state_input = Input(shape=(8, 8, 13), name="state")
    action_input = Input(shape=(8, 8, 13), name="action")

    x = layers.Concatenate()([state_input, action_input])

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    output = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model