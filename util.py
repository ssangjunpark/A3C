import tensorflow as tf
import numpy as np

from feature_extractor import Feature_Extractor
from policy_model import Policy_Model
from value_model import Value_Model

def create_networks(action_space_size, feature_extractor_conv_sizes, feature_extractor_dense_sizes, policy_dense_sizes, value_dense_sizes):
    feature_extractor1 = Feature_Extractor(feature_extractor_conv_sizes, feature_extractor_dense_sizes).iwantmymodelbro()
    feature_extractor2 = Feature_Extractor(feature_extractor_conv_sizes, feature_extractor_dense_sizes).iwantmymodelbro()
    
    policy_model = Policy_Model(feature_extractor1, action_space_size, policy_dense_sizes)
    value_model = Value_Model(feature_extractor2, value_dense_sizes)

    return policy_model, value_model


def image_transformer(image, new_size):
    image = image / 255.0
    gray_scaled = tf.image.rgb_to_grayscale(image)

    resized = tf.image.resize(gray_scaled, [new_size[0], new_size[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.squeeze(resized).numpy().astype(np.float32)