import tensorflow as tf

class Feature_Extractor:
    def __init__(self, conv_sizes, dense_sizes):
        self.conv_sizes = conv_sizes
        self.dense_sizes = dense_sizes
        
        self.model = tf.keras.Sequential()

        for filters, kernal_size, strides in self.conv_sizes:
            self.model.add(
                tf.keras.layers.Conv2D(filters, kernal_size, strides, activation='relu', kernel_initializer='he_uniform')
            )

        self.model.add(tf.keras.layers.Flatten())

        for units in self.dense_sizes:
            self.model.add(
                tf.keras.layers.Dense(units, activation='relu', kernel_initializer='he_uniform')
            )
    
    def iwantmymodelbro(self):
        return self.model