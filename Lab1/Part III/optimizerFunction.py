''' optimizerFunction.function : function implementing the training step. '''
# Optimizers add additional nodes in the tensorflow graph to compute gradients as well as apply them to the variables involved.
# This could be manually performed using tf.gradients etc, but it is a process that is repeated over and over in all Deep Neural # Networks, so the optimizers hide all the gory details. In addition, optimizers do things other than calculate simple gradients # in order to ensure that convergence happens quickly. "Executing" the optimizer inside a tf.Session hence implements:
# 1) computation of gradients with respect to all the variables in the graph, 
# 2) adjusting the value of the variables using these gradients.
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def function(loss, lr):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    return optimizer