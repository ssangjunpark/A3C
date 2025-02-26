import tensorflow as tf

class Policy_Model:
    def __init__(self, feature_extractor, action_space_size, dense_size):
        self.feature_extractor = feature_extractor
        self.action_space_size = action_space_size
        self.dense_size = dense_size

        self.model = feature_extractor
        self.optimizer = tf.keras.optimizers.RMSprop(0.00025, 0.99, 0.0, 1e-6)

        for units in dense_size:
            self.model.add(
                tf.keras.layers.Dense(units, activation='relu', kernel_initializer='he_uniform')
            )

        # add final softmax layer of size action space
        self.model.add(
            tf.keras.layers.Dense(self.action_space_sizes, activation='softmax')
            )
        
        
    
    def predict(self, states):
        return self.model(states)
    
    @tf.function
    def calculate_gradients(self, actions, states, advantages, reg_const):
        with tf.GradientTape() as tape:

            pi_batch_prediction = self.predict(states)
            chosen_action_indicies = tf.stack([tf.range(actions.shape()[0]), actions], axis=1)
            pi_batch = tf.gather_nd(pi_batch_prediction, chosen_action_indicies)

            # entropy to encourage exploration 
            entropy = -tf.reduce_sum(pi_batch * tf.log(pi_batch), axis=1) 
             
            loss = 1 * tf.log(pi_batch) * advantages + reg_const * entropy
            loss = -tf.reduce_sum(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        return gradients
            








        
    
    