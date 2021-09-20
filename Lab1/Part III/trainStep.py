''' trainStep.function : function that implements the training step. '''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def function(session, optimizer):
    return session.run(optimizer)