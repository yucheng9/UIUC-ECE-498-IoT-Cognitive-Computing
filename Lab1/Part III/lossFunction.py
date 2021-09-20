''' lossFunction.function : function implementing the loss, Z = a(X^t*X) + b^t*X, loss = (Z - y) ** 2. '''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def function(a, X, b, y):
    Z = a * tf.linalg.matmul(tf.transpose(X), X) + tf.linalg.matmul(tf.transpose(b), X)
    loss = (Z - y) ** 2
    return loss