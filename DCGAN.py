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
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """

    def self.__init__(self, params):

        assert len(params[0]) == len(params[1]) and len(params[1]) == len(params[2]), "Please make sure param lists are the same size"
        self.num_filters = params[0]
        self.num_strides = params[1]
        self.dropouts = params[2]

    def build_descriminator(self, inputs, kernel_size = None, reuse = False):

        with tf.variable_scope("discriminator", reuse = reuse):

            conv = inputs

            for ind in range(len(self.num_filters)):

                batch_norm = False if ind == 0 else True

                cur_filter = self.num_filters[ind]
                cur_stride = self.num_strides[ind]
                cur_dropout = self.dropouts[ind]

                conv = tf.layers.batch_normalization(conv)

                conv =  tf.layers.conv2d_transpose(
                    inputs = conv, 
                    filters = cur_filter, 
                    kernel_size = kernel_size, 
                    strides = cur_stride, 
                    activation = leaky_relu, 
                    padding = 'valid"
                    )

                if cur_dropout != 0:
                    conv = tf.nn.dropout(conv1, cur_dropout)

            flat = tf.layers.flatten(conv)
            logits = tf.layers.dense(flat,1)

            return logits

class Generator(object):

    def self.__init__(self, encode_params, decode_params, num_channels = 3):
        assert len(encode_params[0]) - 1) == len(decode_params[0]), "encoder must have one more  layer than decoder"
        assert len(encode_params[0]) == len(encode_params[1]) and len(encode_params[1]) == len(encode_params[2]), "Please make sure param lists are the same size"
        assert len(decode_params[0]) == len(decode_params[1]) and len(decode_params[1]) == len(decode_params[2]), "Please make sure param lists are the same size"

        self.enc_filters = encode_params[0]
        self.enc_strides = encode_params[1]
        self.enc_dropout = encode_params[2]

        self.dec_filters = decode_params[0]
        self.dec_strides = decode_params[1]
        self.enc_dropout = decode_params[2]

        self.num_channels = num_channels 

    def build_generator(self, inputs, kernel_size = None, reuse = False):

        conv = inputs
        
        with tf.variable_scope("generator", reuse = reuse):

            skip_layers = []
            
            for ind in range(len(self.enc_filters)):

                cur_filter = self.enc_filters[ind]
                cur_stride = self.enc_strides[ind]
                cur_dropout = self.enc_dropouts[ind]

                conv = tf.layers.conv2d(
                    inputs = conv, 
                    filters = cur_filter, 
                    kernel_size = kernel_size, 
                    strides = cur_stride, 
                    activation = leaky_relu, 
                    padding = 'valid"
                    )
                
                skip_layers.append(conv)
                if cur_dropout != 0:
                    conv = tf.nn.dropout(conv1, cur_dropout)

            for ind in range(len(self.dec_filters)):
                dec_filter = self.enc_filters[ind]
                cur_stride = self.enc_strides[ind]
                cur_dropout = self.enc_dropouts[ind]

                conv = tf.layers.conv2d_transpose(
                inputs = conv, 
                filters = cur_filter, 
                kernel_size = kernel_size, 
                strides = cur_stride, 
                activation = leaky_relu, 
                padding = 'valid"
                )

                if cur_dropout != 0:
                    conv = tf.nn.dropout(conv1, cur_dropout)

                conv = tf.concat([skip_layers[len(layers) - ind - 2], conv], axis=3)

            output = tf.layers.conv2d(
                inputs = conv,
                filters = self.num_channels,
                kernel_size = 1,
                strides = 1,
                activation=tf.nn.tanh,
            )

            return output



                

            





        



def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        s1 = 7*7*128
        fc1 = tf.layers.dense(z, 1024, activation = tf.nn.relu)
        bn1 = tf.layers.batch_normalization(fc1)
        fc2 = tf.layers.dense(bn1, s1, activation = tf.nn.relu)
        bn2 = tf.layers.batch_normalization(fc2)
        logits = tf.reshape(bn2, [-1, 7,7,128])
        conv1 =  tf.layers.conv2d_transpose(logits, 64, 4, strides = 2, activation = leaky_relu, padding = 'same')
        bn3 = tf.layers.batch_normalization(conv1)
        img =  tf.layers.conv2d_transpose(bn3, 1, 4, strides = 2, activation = tf.nn.tanh, padding = 'same')
        img = tf.reshape(img, [-1, 784])
        return img


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    
    HINT: for the discriminator loss, you'll want to do the averaging separately for
    its two components, and then add them together (instead of averaging once at the very end).
    """
    # TODO: compute D_loss and G_loss
    print (logits_real.shape, logits_fake.shape)
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake), logits = logits_fake))
    D_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) 
    D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =  tf.zeros_like(logits_fake), logits = logits_fake))
    D_loss = D_loss1 + D_loss2

    
    return D_loss, G_loss