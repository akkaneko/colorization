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
        self.globalstep = 0
        self.cur_epoch = 0
        

        
    def build_graph(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.batch_size = self.params['batch_size']

        self.raw_images = tf.placeholder(tf.float32, [None, None, None, 3])
        self.bw_images = tf.image.rgb_to_grayscale(self.raw_images)

        #lr = tf.train.exponential_decay(self.params['lr'], self.global_step, 0.1, 5e5)
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
        tf.summary.scalar('accuracy', self.accuracy)
        
        self.D_loss, self.G_loss = self.gan_loss(disc_true, disc_false)
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G_loss', self.G_loss)

        

        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.G_loss, var_list = generator.var_list)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.D_loss, var_list = discriminator.var_list)

        self.saver = tf.train.Saver()
        self.summaries = tf.summary.merge_all()


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
        D_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real) * 0.9, logits = logits_real)) 
        tf.summary.scalar('D_loss_real', D_loss1)
        
        D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =  tf.zeros_like(logits_fake), logits = logits_fake))
        tf.summary.scalar('D_loss_fake', D_loss2)
        
        D_loss = D_loss1 + D_loss2

        return D_loss, G_loss


    def train(self):
        epochs = self.params['num_epochs']
        writer = tf.summary.FileWriter('./graphs/1', self.sess.graph)

        for i in range(epochs):

            data_gen = self.dataset.generator(self.batch_size)
            self.cur_epoch = i + 1
            print ("Starting Epoch Number %d") % (self.cur_epoch)

            for step, batch in enumerate(data_gen):
                self.globalstep+=1

                feed_dict = {self.raw_images: batch}
                
                self.sess.run(
                    [self.gen_optimizer, self.disc_optimizer],
                    feed_dict = feed_dict)

                summ, acc, g_loss, d_loss = self.sess.run(
                    [self.summaries, self.accuracy, self.G_loss, self.D_loss],
                    feed_dict = feed_dict
                )

                if step % 10 == 0:
                    print "Step: %d Accuracy: %f, GenLoss: %f, DisLoss: %f" % (step, acc, g_loss, d_loss)
                    writer.add_summary(summ, self.globalstep)


                if step % 100 == 0:
                    self.sample(feed_dict)
          
            self.saver.save(self.sess, self.params['save_path'], write_meta_graph=False)
            
        
    def load(self):
        ckpt = self.params['load_path']
        if ckpt is not None:
            self.saver.restore(self.sess, ckpt)
            return True
        return False
    
    def sample(self, feed_dict):
        bw_image, fake_image, real_image = self.sess.run([self.bw_images, self.colorized, self.color_images], feed_dict)
        image = (np.squeeze(np.array(fake_image)) + 1) / 2
        original = (np.squeeze(np.array(real_image)) + 1) / 2
        bw = (np.squeeze(np.array(bw_image)))
        
        bw = Image.fromarray((bw[0]).astype(np.uint8))
        plt.imshow(bw, interpolation='none')
        plt.show()
        
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

