import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation
from keras.optimizers import RMSprop, SGD
from keras import regularizers


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 12, name=name+"_block"+str(i+1))
    return x

def conv_block(x, growth_rate, name):
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name+"_0_bn")(x)
    x1 = Activation("relu", name=name+"_0_relu")(x1)
    x1 = Conv2D(4*growth_rate, 1, use_bias=False,
                name=name+"_1_conv")(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name+"_1_bn")(x1)
    x1 = Activation("relu", name=name+"_1_relu")(x1)
    x1 = Conv2D(growth_rate, 3, padding="same",use_bias=False,
                name=name+"_2_conv")(x1)
    x = Concatenate(axis=bn_axis, name=name+"_concat")([x, x1])
    return x

def transition_block(x, reduction, name):
    bn_axis = 3
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name+"_bn")(x)
    x = Activation("relu", name=name+"_relu")(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis]*reduction), 1, use_bias=False,
               name=name+"_conv")(x)
    x = AveragePooling2D(2, strides=2, name=name+"_pool")(x)
    return x

def DenseID(input_tensor=None,
            input_shape=[64, 64, 3],
            classes=5749):
    bn_axis = 3
    denseid = 2048
    blocks = [12, 12, 12, 32]
    input_tensor = Input(input_shape)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name="bn0")(input_tensor)
    x = Activation("relu", name="relu0")(x)
    x = MaxPooling2D(3, strides=2, padding="same", name="pool0")(x)

    x = dense_block(x, blocks[0], name="block_1")
    x = transition_block(x, 0.5, name="trans_1")
    x = dense_block(x, blocks[1], name="block_2")
    x = transition_block(x, 0.5, name="trans_2")
    x = dense_block(x, blocks[2], name="block_3")
    # x = transition_block(x, 0.5, name="trans_3")
    # x = dense_block(x, blocks[3], name="block_4")

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name="bn")(x)

    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(denseid, activation="relu", name="denseid")(x)
    x = Dense(classes, activation="softmax", name="fc")(x)

    model = Model(input_tensor, x, name="DenseID")
    return model