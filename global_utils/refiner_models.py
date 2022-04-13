"""
Defines networks.

@Encoder_resnet
@Encoder_fc3_dropout

Helper:
@get_encoder_fn_separate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_fc3_dropout(nn.Module):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    def __init__(self, num_input=170, num_output=85):
        super(Encoder_fc3_dropout, self).__init__()
        #if reuse:
        #    print('Reuse is on!')
        self.fc1 = nn.Linear(num_input, 1024)  # slim.fully_connected(x, 1024, scope='fc1')
        self.dropout1 = nn.Dropout(0.5)   # slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)  # slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        #small_xavier = variance_scaling_initializer(
        #    factor=.01, mode='FAN_AVG', uniform=True)
        self.fc3 =  nn.Linear(1024, num_output) #slim.fully_connected(net,num_output,activation_fn=None,weights_initializer=small_xavier,scope='fc3')
        nn.init.xavier_uniform_(self.fc3.weight)

        #variables = tf.contrib.framework.get_variables(scope)
        #return net, variables

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def get_encoder_fn_separate(model_type):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet' in model_type:
        encoder_fn = Encoder_resnet
    else:
        print('Unknown encoder %s!' % model_type)
        exit(1)

    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout

    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        import ipdb
        ipdb.set_trace()

    return encoder_fn, threed_fn
