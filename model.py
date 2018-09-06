import tensorflow as tf
import numpy as np
from data import Cifar10Dataset
from DCGAN import Generator, Discriminator
from abc import abstractmethod
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image




class Colorize(object):

    def __init__(self, sess, params):

        self.sess = sess
        self.params = params
        self.create_dataset(training = params['training'])

        
    def build_graph(self):

        self.raw_images = tf.placeholder(tf.float32, [None, None, None, 3])
        self.bw_images = tf.image.rgb_to_grayscale(self.raw_images)

        lr = self.params['lr']
        thresh = self.params['threshold']

        ## Center around 0 ##
        self.color_images =  (self.raw_images / 255.0) * 2 - 1

        generator = self.create_generator()
        discriminator = self.create_discriminator()

        ## Generated image: [batch_size, h, w, channels]
        self.colorized = generator.generate(
            self.bw_images, 
            kernel_size = self.params['kernel_size']
        )

        ## Discriminated logits: [?, ?, ?, ?, 1]
        disc_false = discriminator.discriminate(
            tf.concat(axis = 3, values = [self.colorized, self.bw_images]),
            kernel_size = self.params['kernel_size']
        )

        ## Discriminated logits: [?, ?, ?, ?, 1]
        disc_true  = discriminator.discriminate(
            tf.concat(axis = 3, values = [self.color_images, self.bw_images]),
            kernel_size = self.params['kernel_size'],
            reuse = True)

        self.accuracy = self.calculate_accuracy(self.color_images, self.colorized, thresh)
        self.D_loss, self.G_loss = self.gan_loss(disc_true, disc_false)

        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.G_loss)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.D_loss)

        self.saver = tf.train.Saver()


    def calculate_accuracy(self, generated, original, threshold):

        diff_r = tf.abs(tf.round(generated[..., 0]) - tf.round(original[..., 0]))
        diff_g = tf.abs(tf.round(generated[..., 1]) - tf.round(original[..., 1]))
        diff_b = tf.abs(tf.round(generated[..., 2]) - tf.round(original[..., 2]))

        predr = tf.cast(tf.less_equal(diff_r, threshold), tf.float64)      
        predg = tf.cast(tf.less_equal(diff_g, threshold), tf.float64)      
        predb = tf.cast(tf.less_equal(diff_b, threshold), tf.float64)   

        pred = predr * predg * predb

        return tf.reduce_mean(pred)


    def gan_loss(self, logits_real, logits_fake):

        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake), logits = logits_fake))
        D_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) 
        D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =  tf.zeros_like(logits_fake), logits = logits_fake))
        D_loss = D_loss1 + D_loss2

        return D_loss, G_loss


    def train(self):
        epochs = self.params['num_epochs']
        batch_size = self.params['batch_size']

        data_gen = self.dataset.generator(batch_size)

        for i in range(epochs):
            
            self.cur_epoch = i
            print ("Starting Epoch Number %d") % (self.cur_epoch)

            for step, batch in enumerate(data_gen):

                feed_dict = {self.raw_images: batch}
                
                self.sess.run(
                    [self.gen_optimizer, self.disc_optimizer],
                    feed_dict = feed_dict)

                acc, g_loss, d_loss = self.sess.run(
                    [self.accuracy, self.G_loss, self.D_loss],
                    feed_dict = feed_dict
                )

                if step % 5 == 0:
                    print "Step: %d Accuracy: %f, GenLoss: %f, DisLoss: %f" % (step, acc, g_loss, d_loss)

                if step % 50 == 0:
                    self.sample(feed_dict)

    def sample(self, feed_dict):
        fake_image, real_image = self.sess.run([self.colorized, self.color_images], feed_dict)
        image = (np.squeeze(np.array(fake_image)) + 1) / 2
        original = (np.squeeze(np.array(real_image)) + 1) / 2
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        plt.imshow(image, interpolation='none')
        plt.show()
        original = Image.fromarray((original[0] * 255).astype(np.uint8))
        plt.imshow(original, interpolation='none')
        plt.show()


    @abstractmethod
    def create_generator(self):
        pass

    @abstractmethod
    def create_discriminator(self):
        pass

    @abstractmethod
    def create_dataset(self, training = True):
        pass

class Color_CIFAR(Colorize):

    def __init__(self, datapath, sess, params):
        self.path = datapath
        super(Color_CIFAR, self).__init__(sess, params)

    def create_dataset(self, training = True):
        self.dataset = Cifar10Dataset(self.path, training)

    def create_generator(self):
        enc_filters = [64, 128, 256, 512, 512]
        enc_strides = [1, 2, 2, 2, 2]
        enc_dropout = [0, 0, 0, 0, 0]
        encode_params = (enc_filters, enc_strides, enc_dropout)

        dec_filters = [512, 256, 128, 64]
        dec_strides = [2, 2, 2, 2]
        dec_dropout = [0.5, 0.5, 0, 0]
        decode_params = (dec_filters, dec_strides, dec_dropout)

        return Generator(encode_params, decode_params)

    def create_discriminator(self):
        filters = [64, 128, 256, 512]
        strides = [2, 2, 2, 1]
        dropout = [0, 0, 0, 0]
        params = (filters, strides, dropout)
        
        return Discriminator(params)

