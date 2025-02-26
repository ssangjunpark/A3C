import tensorflow as tf
import numpy as np

from feature_extractor import Feature_Extractor
from policy_model import Policy_Model
from value_model import Value_Model

def create_networks(action_space_size, feature_extractor_conv_sizes, feature_extractor_dense_sizes, policy_dense_sizes, value_dense_sizes):
    fe = Feature_Extractor(feature_extractor_conv_sizes, feature_extractor_dense_sizes)

    policy_model = Policy_Model(fe, action_space_size, policy_dense_sizes) 
    value_model = Value_Model(fe, value_dense_sizes)

    return policy_model, value_model


def image_transformer(image, new_size):
    gray_scaled = tf.image.rgb_to_grayscale(image)

    resized = tf.image.resize(gray_scaled, [new_size[0], new_size[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.squeeze(resized).numpy().astype(np.float32)