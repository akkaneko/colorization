import tensorflow as tf
import numpy as np

class Discriminator(object):

    """
    Discriminator
    
    Inputs:
    - input: inputs
    - params: tuple of the form (num_filters, num_strides, dropout, kernel_size)
              where num_filters and num_strides are lists, and kernel_size is an int.

              num_filters and num_strides must be the same length. 
    
    Returns:
    Logits from Discriminator
    """

    def __init__(self, params):

        assert len(params[0]) == len(params[1]) and len(params[1]) == len(params[2]), "Please make sure param lists are the same size"
        self.num_filters = params[0]
        self.num_strides = params[1]
        self.dropouts = params[2]

    def discriminate(self, inputs, kernel_size = None, reuse = False):

        with tf.variable_scope("discriminator", reuse = reuse):

            conv = inputs + tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=0.1, dtype=tf.float32)

            for ind in range(len(self.num_filters)):

                batch_norm = False if ind == 0 else True

                cur_filter = self.num_filters[ind]
                cur_stride = self.num_strides[ind]
                cur_dropout = self.dropouts[ind]

                conv =  tf.layers.conv2d(
                    inputs = conv, 
                    filters = cur_filter, 
                    kernel_size = kernel_size, 
                    strides = cur_stride, 
                    padding = "same"
                    )

                if batch_norm:
                    conv = tf.layers.batch_normalization(conv, training = True)
                
                conv = tf.nn.leaky_relu(conv)

                if cur_dropout != 0:
                    conv = tf.nn.dropout(conv, cur_dropout)

            

            logits = tf.layers.conv2d(
                inputs = conv,
                filters = 1,
                kernel_size = 4,
                strides = 1,
                padding = "same"
            )
            
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            return logits

class Generator(object):

    def __init__(self, encode_params, decode_params, num_channels = 3):
        assert (len(encode_params[0]) - 1 == len(decode_params[0])), "encoder must have one more layer than decoder"
        assert len(encode_params[0]) == len(encode_params[1]) and len(encode_params[1]) == len(encode_params[2]), "Please make sure param lists are the same size"
        assert len(decode_params[0]) == len(decode_params[1]) and len(decode_params[1]) == len(decode_params[2]), "Please make sure param lists are the same size"

        self.enc_filters = encode_params[0]
        self.enc_strides = encode_params[1]
        self.enc_dropout = encode_params[2]

        self.dec_filters = decode_params[0]
        self.dec_strides = decode_params[1]
        self.dec_dropout = decode_params[2]

        self.num_channels = num_channels 

    def generate(self, inputs, kernel_size = None, reuse = tf.AUTO_REUSE):
        
        with tf.variable_scope("generator", reuse = reuse):

            conv = inputs
            skip_layers = []
            
            for ind in range(len(self.enc_filters)):

                cur_filter = self.enc_filters[ind]
                cur_stride = self.enc_strides[ind]
                cur_dropout = self.enc_dropout[ind]

                conv = tf.layers.conv2d(
                    inputs = conv, 
                    filters = cur_filter, 
                    kernel_size = kernel_size, 
                    strides = cur_stride, 
                    padding = "same"
                    )

                conv = tf.layers.batch_normalization(conv, training = True)
                conv = tf.nn.leaky_relu(conv)

                skip_layers.append(conv)
                if cur_dropout != 0:
                    conv = tf.nn.dropout(conv, cur_dropout)

            for ind in range(len(self.dec_filters)):
                cur_filter = self.dec_filters[ind]
                cur_stride = self.dec_strides[ind]
                cur_dropout = self.dec_dropout[ind]

                conv = tf.layers.conv2d_transpose(
                inputs = conv, 
                filters = cur_filter, 
                kernel_size = kernel_size, 
                strides = cur_stride, 
                padding = "same"
                )

                conv = tf.layers.batch_normalization(conv, training = True)
                conv = tf.nn.relu(conv)

                if cur_dropout != 0:
                    conv = tf.nn.dropout(conv, cur_dropout)

                conv = tf.concat([skip_layers[len(skip_layers) - ind - 2], conv], axis=3)

            output = tf.layers.conv2d(
                inputs = conv,
                filters = self.num_channels,
                kernel_size = 1,
                strides = 1,
                activation=tf.nn.tanh,
                padding = "same"
            )
            
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            return output

        


