''' computeLoss.function : function that provides a printable value of the loss. By printable, what we mean
    is that the value of loss is visible on using python print(). Using print directly on tensorflow variables
    or constants doesn't show their value. '''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def function(session, loss):
    return loss.eval(session=session)[0][0]