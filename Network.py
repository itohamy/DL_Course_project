import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from tensorflow.contrib import layers
from tensorflow.contrib.learn import ModeKeys
import keras.backend as K
import tensorflow as tf
import layers
import acts
BN_EPSILON = 0.001



class FusionNet(object):
    def __init__(self):
        self.act_fn = acts.pRelu
        self.kernel_num = 32
        self.output_dim = 3
        self.log = 1
        print("FusionNet Loading"),

    def skip_connection(self, input_, output_):
        return tf.add(input_, output_)

    def res_block_with_n_conv_layers(self, input_, output_dim, num_repeat, name="res_block"):
        output_ = layers.conv2d_same_repeat(input_, output_dim,
                                            num_repeat=num_repeat, activation_fn=self.act_fn, name=name)
        return self.skip_connection(input_, output_)

    def res_block_with_3_conv_layers(self, input_, output_dim, name="res_block"):
        return self.res_block_with_n_conv_layers(input_, output_dim, num_repeat=3, name=name)

    def conv_res_conv_block(self, input_, output_dim, name="conv_res_conv_block"):
        with tf.variable_scope(name):
            conv1 = layers.conv2d_same_act(input_, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv1")
            res = self.res_block_with_3_conv_layers(conv1, output_dim, name="res_block")
            conv2 = layers.conv2d_same_act(res, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv2")
            return conv2

    def encoder(self, input_):
        self.down1 = self.conv_res_conv_block(input_, self.kernel_num, name="down1")
        pool1 = layers.max_pool(self.down1, name="pool1")

        self.down2 = self.conv_res_conv_block(pool1, self.kernel_num * 2, name="down2")
        pool2 = layers.max_pool(self.down2, name="pool2")

        self.down3 = self.conv_res_conv_block(pool2, self.kernel_num * 4, name="down3")
        pool3 = layers.max_pool(self.down3, name="pool3")

        self.down4 = self.conv_res_conv_block(pool3, self.kernel_num * 8, name="down4")
        pool4 = layers.max_pool(self.down4, name="pool4")

        if self.log == 1:
            print("encoder input : ", input_.get_shape())
            print("conv1 : ", self.down1.get_shape())
            print("pool1 : ", pool1.get_shape())
            print("conv2 : ", self.down2.get_shape())
            print("pool2 : ", pool2.get_shape())
            print("conv3 : ", self.down3.get_shape())
            print("pool3 : ", pool3.get_shape())
            print("conv4 : ", self.down4.get_shape())
            print("pool4 : ", pool4.get_shape())

        return pool4

    def decoder(self, input_):

        conv_trans4 = layers.conv2dTrans_same_act(input_, self.down4.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool4")
        res4 = self.skip_connection(conv_trans4, self.down4)
        up4 = self.conv_res_conv_block(res4, self.kernel_num * 8, name="up4")

        conv_trans3 = layers.conv2dTrans_same_act(up4, self.down3.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool3")
        res3 = self.skip_connection(conv_trans3, self.down3)
        up3 = self.conv_res_conv_block(res3, self.kernel_num * 4, name="up3")

        conv_trans2 = layers.conv2dTrans_same_act(up3, self.down2.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool2")
        res2 = self.skip_connection(conv_trans2, self.down2)
        up2 = self.conv_res_conv_block(res2, self.kernel_num * 2, name="up2")

        conv_trans1 = layers.conv2dTrans_same_act(up2, self.down1.get_shape(),
                                                 activation_fn=self.act_fn, with_logit=False, name="unpool1")
        res1 = self.skip_connection(conv_trans1, self.down1)
        up1 = self.conv_res_conv_block(res1, self.kernel_num, name="up1")

        if self.log == 1:
            print("dncoder input : ", input_.get_shape())
            print("convT1 : ", conv_trans4.get_shape())
            print("res1 : ", res4.get_shape())
            print("up1 : ", up4.get_shape())
            print("convT2 : ", conv_trans3.get_shape())
            print("res2 : ", res3.get_shape())
            print("up2 : ", up3.get_shape())
            print("convT3 : ", conv_trans2.get_shape())
            print("res3 : ", res2.get_shape())
            print("up3 : ", up2.get_shape())
            print("convT4 : ", conv_trans1.get_shape())
            print("res4 : ", res1.get_shape())
            print("up4 : ", up1.get_shape())

        return up1

    def inference(self, input_):
        encode_vec = self.encoder(input_)
        bridge = self.conv_res_conv_block(encode_vec, self.kernel_num * 16, name="bridge")
        decode_vec = self.decoder(bridge)
        output = layers.bottleneck_layer(decode_vec, self.output_dim, name="output")

        if self.log == 1:
            print("output : ", output.get_shape())

        print("Complete!!")

        return output    




class Network():
    training = tf.placeholder(tf.bool)
    def __init__(self):
        pass


    def build(self, input_batch, example_parameter_string='Example', example_parameter_int = 3):
        new_input = tf.reshape(input_batch, [-1, 128, 128, 1])
        fusionNet = FusionNet()
        out = fusionNet.inference(new_input)
        
        print("out:", out.shape)   
        return tf.reshape(out,[-1,128,128,3])
    
    
    
"""    
  
        This function is where you write the code for your network. The input is a batch of images of size (N,H,W,1)
        N is the batch size
        H is the image height
        W is the image width
        The output needs to be of shape (N,H,W,3)
        where the last channel is the UNNORMALIZEd calss probabilities (before softmax) for classes background,
        foreground and edge.
        :param input_batch:
        :param example_parameter: A parameter for example. See Params file to change the value
        :return:
        
        for i in range(example_parameter_int):
            print('{}: '.format(i) + example_parameter_string)

        ########### new: ########
        
        # pool1 (10, 64, 64, 32) * 
        
        new_input = tf.reshape(input_batch, [-1, 256, 256, 1])
        conv1 = tf.layers.conv2d(
                inputs=new_input,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv11 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv11, pool_size=[2, 2], strides=2)
        print("conv11", conv11.shape)
        print('pool1', pool1.shape)
        
        # pool2 (10, 32, 32, 64)
        
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv22 = tf.layers.conv2d(
                inputs=conv2,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2)
        print("conv22", conv22.shape)
        print("pool2", pool2.shape)
        
        # pool3 (10, 16, 16, 128)
        
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv33 = tf.layers.conv2d(
                inputs=conv3,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2, 2], strides=2)
        print("conv33", conv33.shape)
        print("pool3", pool3.shape)
        
        # pool4 (10, 8, 8, 256)
        
        conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv44 = tf.layers.conv2d(
                inputs=conv4,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv44, pool_size=[2, 2], strides=2)
        print("conv44", conv44.shape)
        print("pool4", pool4.shape)
        
        # dropout (10, 8, 8, 512)
        
        conv5 = tf.layers.conv2d(
                inputs=pool4,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv55 = tf.layers.conv2d(
                inputs=conv5,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        
        #dropout = tf.layers.dropout(inputs=conv55, rate=0.5)
        #print("dropout", dropout.shape)
        
        # up6 (10, 16, 16, 768)
        
        upsample6 = tf.image.resize_images(
                images = conv55,
                size = [32, 32],
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
        up6 = tf.concat([upsample6,conv44],3)
        print("up6", up6.shape)
        
        # conv66 (10, 16, 16, 256)
        conv6 = tf.layers.conv2d(
                inputs=up6,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv66 = tf.layers.conv2d(
                inputs=conv6,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        print("conv66", conv66.shape)
        
        # up7 (10, 32, 32, 384)
        
        upsample7 = tf.image.resize_images(
                images = conv66,
                size = [64, 64],
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
        up7 = tf.concat([upsample7,conv33],3)
        print("up7", up7.shape)
        
        
        # conv77 
        conv7 = tf.layers.conv2d(
                inputs=up7,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv77 = tf.layers.conv2d(
                inputs=conv7,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        print("conv77", conv77.shape)
        
        # up8 
        
        upsample8 = tf.image.resize_images(
                images = conv77,
                size = [128, 128],
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
        up8 = tf.concat([upsample8,conv22],3)
        print("up8", up8.shape)
        
        
        # conv88
        conv8 = tf.layers.conv2d(
                inputs=up8,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv88 = tf.layers.conv2d(
                inputs=conv8,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        print("conv88", conv88.shape)
        
        # up9 
        
        upsample9 = tf.image.resize_images(
                images = conv88,
                size = [256, 256],
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
        up9 = tf.concat([upsample9,conv11],3)
        print("up9", up9.shape)
        
        
        # conv99
        conv9 = tf.layers.conv2d(
                inputs=up9,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        conv99 = tf.layers.conv2d(
                inputs=conv9,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        print("conv99", conv99.shape)
        
#        conv10 = tf.layers.conv2d(
#                inputs=conv99,
#                filters=1,
#                kernel_size=[3, 3],
#                padding="same",
#                activation=tf.nn.relu)
#        print("conv10", conv10.shape)

#        conv11 = tf.layers.conv2d(
#                inputs=conv10,
#                filters=8,
#                kernel_size=[3, 3],
#                padding="same",
#                activation=tf.nn.relu)
#        print("conv11", conv11.shape)
#
#        conv12 = tf.layers.conv2d(
#                inputs=conv11,
#                filters=4,
#                kernel_size=[3, 3],
#                padding="same",
#                activation=tf.nn.relu)
#        print("conv11", conv12.shape)


        out = tf.layers.dense(inputs=up9, units=(3))
        print("out",out.shape)
"""
