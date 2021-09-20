''' initializeX.function : function returning the tensorflow variable X initialized to a random value. '''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def function(shape):
    weights = tf.Variable(np.random.rand(shape[0], shape[1]), dtype=tf.float32)
    return weights
                        