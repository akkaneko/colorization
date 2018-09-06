import tensorflow as tf
import numpy as np
from data import Cifar10Dataset
from DCGAN import Generator, Discriminator

class Colorize(object):

    def __init__(self, sess, params):

        self.sess = sess
        self.params = params
        
    def build_graph(self, sess):

        self.raw_images = tf.placeholder(tf.float32, [None, None, None, 3])
        self.bw_images = tf.image.rgb_to_grayscale(raw_images)

        lr = params['lr']

        ## Center around 0 ##
        self.color_images =  (self.raw_images / 255.0) * 2 - 1

        generator = self.create_generator()
        discriminator = self.create_discriminator()

        colorized = generator.build_graph(
            self.bw_images, 
            kernel_size = params['kernel_size'], 
            reuse = params['reuse'])

        disc_false = discriminator.build_graph(
            tf.concat(3, [colorized, slef.bw_images]),
            kernel_size = params['kernel_size']
        )

        disc_true  = discriminator.build_graph(
            tf.concat(3, [self.color_images, slef.bw_images]),
            kernel_size = params['kernel_size'],
            reuse = True

        self.D_loss, self.G_loss = self.gan_loss(disc_true, disc_false)

        gen_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.G_loss)
        disc_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.D_loss)

    def gan_loss(self, logits_real, logits_fake):
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
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake), logits = logits_fake))
        D_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) 
        D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =  tf.zeros_like(logits_fake), logits = logits_fake))
        D_loss = D_loss1 + D_loss2

        return D_loss, G_loss

        


    @abstractmethod
    def create_generator(self):
        pass

    @abstractmethod
    def create_discriminator(self):
        pass



    def train(self, sess):


