import tensorflow as tf

class Value_Model:
    def __init__(self, feature_extractor, dense_size):
        self.dense_size = dense_size

        self.model = feature_extractor
        self.optimizer = tf.keras.optimizers.RMSprop(0.00025, 0.99, 0.0, 1e-6)

        for units in dense_size:
            self.model.add(
                tf.keras.layers.Dense(units, activation='relu', kernel_initializer='he_uniform')
            )

        # final regression layer lol
        self.model.add(
            tf.keras.layers.Dense(1, activation=None)
        )

    def predict(self, states):
        return self.model(states)
    
    @tf.function
    def calculate_gradients(self, states, returns):
        with tf.GradientTape() as tape:
            
            values = self.predict(states)

            loss = tf.reduce_sum(((returns - values) ** 2))

        gradients = tape.gradient(loss, self.model.trainable_variables)

        return gradients

        