from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size=128,
                 batch_size=1, sample_size=1, output_size=128,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_B = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B_128 = tf.image.resize_images(self.real_B, (int(self.image_size / 2), int(self.image_size / 2)))
        self.real_B_64 = tf.image.resize_images(self.real_B, (int(self.image_size / 4), int(self.image_size / 4)))
        self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.real_A_128 = tf.image.resize_images(self.real_A, (int(self.image_size / 2), int(self.image_size / 2)))
        self.real_A_64 = tf.image.resize_images(self.real_A, (int(self.image_size / 4), int(self.image_size / 4)))

        self.fake_B_64 = self.generator_128_to_64(self.real_A)
        self.fake_B_128 = self.generator_64_to_128(self.fake_B_64)
        self.fake_B_256 = self.generator_128_to_256(self.fake_B_128)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB_64 = tf.concat([self.real_A_64, self.fake_B_64])
        self.fake_AB_128 = tf.concat([self.real_A_128, self.fake_B_128])
        self.fake_AB_256 = tf.concat([self.real_A, self.fake_B_256], 3)
        self.D, self.D_logits = self.discriminator_256(self.real_AB, reuse=False)
        self.D_64, self.D_logits_64 = self.discriminator_64(self.fake_AB_64, reuse=True)
        self.D_128, self.D_logits_128 = self.discriminator_128(self.fake_AB_128, reuse=True)
        self.D_256, self.D_logits_256 = self.discriminator_256(self.fake_AB_256, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d_64_sum = tf.summary.histogram("d_64", self.D_64)
        self.d_128_sum = tf.summary.histogram("d_128", self.D_128)
        self.d_256_sum = tf.summary.histogram("d_256", self.D_256)
        self.fake_B_64_sum = tf.summary.image("fake_B_64", self.fake_B_64)
        self.fake_B_128_sum = tf.summary.image("fake_B_128", self.fake_B_128)
        self.fake_B_256_sum = tf.summary.image("fake_B_256", self.fake_B_256)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, 
                                                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_64, 
                                                                                        labels=tf.zeros_like(self.D_64)))
        self.d_loss_fake_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_128, 
                                                                                        labels=tf.zeros_like(self.D_128)))
        self.d_loss_fake_256 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_256, 
                                                                                    labels=tf.zeros_like(self.D_256)))
        self.mask = tf.greater(tf.abs(self.real_B - self.fake_B_256), 0) # only calculate the mean of different part
        self.g_loss_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_64, labels=tf.ones_like(self.D_64))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B_64 - self.fake_B_64))\
                        + self.L1_lambda * 2 * tf.abs(tf.reduce_mean(self.real_B_64) - tf.reduce_mean(tf.abs(self.real_B_64 - self.fake_B_64)))
        self.g_loss_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_128, labels=tf.ones_like(self.D_128))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B_128 - self.fake_B_128))\
                        + self.L1_lambda * 2 * tf.abs(tf.reduce_mean(self.real_B_128) - tf.reduce_mean(tf.abs(self.real_B_128 - self.fake_B_128)))
        self.g_loss_256 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_256, labels=tf.ones_like(self.D_256))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B_256))\
                        + self.L1_lambda * 2 * tf.abs(tf.reduce_mean(self.real_B) - tf.reduce_mean(tf.abs(self.real_B - self.fake_B_256)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_64_sum = tf.summary.scalar("d_loss_fake_64", self.d_loss_fake_64)
        self.d_loss_fake_128_sum = tf.summary.scalar("d_loss_fake_128", self.d_loss_fake_128)
        self.d_loss_fake_256_sum = tf.summary.scalar("d_loss_fake_256", self.d_loss_fake_256)

        self.d_loss_64 = self.d_loss_real + self.d_loss_fake_64
        self.d_loss_128 = self.d_loss_real + self.d_loss_fake_128
        self.d_loss_256 = self.d_loss_real + self.d_loss_fake_256

        self.g_loss_64_sum = tf.summary.scalar("g_loss_64", self.g_loss_64)
        self.g_loss_128_sum = tf.summary.scalar("g_loss_128", self.g_loss_128)
        self.g_loss_sum_256 = tf.summary.scalar("g_loss_256", self.g_loss_256)
        self.d_loss_64_sum = tf.summary.scalar("d_loss_64", self.d_loss_64)
        self.d_loss_128_sum = tf.summary.scalar("d_loss_128", self.d_loss_128)
        self.d_loss_256_sum = tf.summary.scalar("d_loss_256", self.d_loss_256)

        t_vars = tf.trainable_variables()

        self.d_vars_64 = []
        self.d_vars_128 = []
        self.d_vars_256 = []
        for var in t_vars:
            if 'd_' in var.name:
                if '64' in var.name:
                    self.d_vars_64.append(var)
                if '128' in var.name:
                    self.d_vars_128.append(var)
                else:
                    self.d_vars_256.append(var)
        self.g_vars_64 = []
        self.g_vars_128 = []
        self.g_vars_256 = []
        for var in t_vars:
            if 'g_' in var.name:
                if '64' in var.name:
                    self.g_vars_64.append(var)
                if '128' in var.name:
                    self.g_vars_128.append(var)
                else:
                    self.g_vars_256.append(var)

        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim_64 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss_64, var_list=self.d_vars_64)
        g_optim_64 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss_64, var_list=self.g_vars_64)
        d_optim_128 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss_128, var_list=self.d_vars_128)
        g_optim_128 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss_128, var_list=self.g_vars_128)
        d_optim_256 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss_256, var_list=self.d_vars_256)
        g_optim_256 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss_256, var_list=self.g_vars_256)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum_64 = tf.summary.merge([self.d_64_sum,
            self.fake_B_64_sum, self.d_loss_fake_64_sum, self.g_loss_64_sum])
        self.d_sum_64 = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_64_sum])
        self.g_sum_128 = tf.summary.merge([self.d_128_sum,
            self.fake_B_128_sum, self.d_loss_fake_128_sum, self.g_loss_128_sum])
        self.d_sum_128 = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_128_sum])
        self.g_sum_256 = tf.summary.merge([self.d_256_sum,
            self.fake_B_256_sum, self.d_loss_fake_256_sum, self.g_loss_256_sum])
        self.d_sum_256 = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_256_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            #np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim_64, self.d_sum_64],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim_64, self.g_sum_64],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update D network
                _, summary_str = self.sess.run([d_optim_128, self.d_sum_128],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim_128, self.g_sum_128],
                                               feed_dict={ self.real_data: batch_images })

                # Update D network
                _, summary_str = self.sess.run([d_optim_256, self.d_sum_256],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim_256, self.g_sum_256],
                                               feed_dict={ self.real_data: batch_images })

                errD_fake_64 = self.d_loss_fake_64.eval({self.real_data: batch_images})
                errD_fake_128 = self.d_loss_fake_128.eval({self.real_data: batch_images})
                errD_fake_256 = self.d_loss_fake_256.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG_64 = self.g_loss_64.eval({self.real_data: batch_images})
                errG_128 = self.g_loss_128.eval({self.real_data: batch_images})
                errG_256 = self.g_loss_256.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: [%.8f, %.8f, %.8f], g_loss: [%.8f, %.8f, %.8f]" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, 
                        errD_fake_64+errD_real, errD_fake_128+errD_real, errD_fake_256+errD_real,
                        errG_64, errG_128, errG_256))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 500) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator_64(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator_64") as scope:

            # image is 64 x 64 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (32 x 32 x self.df_dim)
            h1 = lrelu(self.d_bn3(conv2d(h1, self.df_dim*2, d_h=1, d_w=1, name='d_h1_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h2 = linear(tf.reshape(h1, [self.batch_size, -1]), 1, 'd_h1_lin')

            return tf.nn.sigmoid(h2), h2

    def discriminator_128(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator_128") as scope:

            # image is 128 x 128 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (64 x 64 x self.df_dim)
            h1 = lrelu(self.d_bn2(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h2 = lrelu(self.d_bn3(conv2d(h1, self.df_dim*4, d_h=1, d_w=1, name='d_h2_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h2_lin')

            return tf.nn.sigmoid(h3), h3

    def discriminator_256(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator_256") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def residual_block(self, image, sur_name):
        rb1 = self.batch_norm(conv2d(image, self.gf_dim * 4, name='g_rb1_conv' + sur_name), name='g_bn_rg_1' + sur_name)
        rb1 = tf.nn.relu(rb1)
        rb2 = self.batch_norm(conv2d(rb1, self.gf_dim * 4, name='g_rb2_conv' + sur_name), name='g_bn_rg_2' + sur_name)
        rb_sum = tf.add(image, rb2, name='g_rb_add' + sur_name)
        return tf.nn.relu(rb_sum)

    def generator_64_to_128(self, image, y=None):
        with tf.variable_scope("generator_64_to_128") as scope:
            rb1 = self.residual_block(image, sur_name='rb1')
            rb2 = self.residual_block(rb1, sur_name='rb2')
            s = self.output_size
            s2 = int(s/2)
            self.d7, self.d7_w, self.d7_b = deconv2d(rb2,
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            # d7 is (128 x 128 x self.gf_dim*1*2)
            return tf.nn.relu(self.d7)

    def generator_128_to_256(self, image, y=None):
        with tf.variable_scope("generator_128_to_256") as scope:
            rb1 = self.residual_block(image, sur_name='rb1')
            rb2 = self.residual_block(rb1, sur_name='rb2')
            s = self.output_size
            self.d8, self.d8_w, self.d8_b = deconv2d(rb2,
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.relu(self.d8)

    def generator_128_to_64(self, image, y=None):
        with tf.variable_scope("generator_128_to_64") as scope:

            s = self.output_size
            s4, s8, s16, s32, s64, s128 = int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (128 x 128 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (64 x 64 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (32 x 32 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (16 x 16 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (8 x 8 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (4 x 4 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (2 x 2 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e7),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = self.g_bn_d1(self.d1)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = self.g_bn_d2(self.d2)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = self.g_bn_d3(self.d3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            return tf.nn.relu(self.d6)

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files_all = glob('./datasets/{}/val_test/*.jpg'.format(self.dataset_name))

        # sort testing input
        # n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        # sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        max_size = 10000
        batch_count = 0
        while len(sample_files_all) > max_size * batch_count:
            endIdx = max_size * (batch_count + 1)
            if (len(sample_files_all) < endIdx):
                endIdx = len(sample_files_all)
            sample_files = sample_files_all[batch_count * max_size : endIdx]
            # load testing input
            print("Loading testing images ... from {0} to {1} of total {2}".format(batch_count * max_size, endIdx, len(sample_files_all)))
            batch_count += 1
            sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

            if (self.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)

            sample_images = [sample_images[i:i+self.batch_size]
                            for i in xrange(0, len(sample_images), self.batch_size)]
            sample_images = np.array(sample_images)
            print(sample_images.shape)

            start_time = time.time()
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            print("file number: {}".format(len(sample_files)))

            for i, sample_image in enumerate(sample_images):
                idx = i
                fileName = sample_files[idx].split('/')[-1].split('.jpg')[0]
                print("sampling image {}, {} of total {}".format(fileName, idx + (batch_count - 1) * max_size, len(sample_files_all)))
                samples = self.sess.run(
                    self.fake_B_sample,
                    feed_dict={self.real_data: sample_image}
                )
                save_images(samples, [self.batch_size, 1],
                            './{}/{}.png'.format(args.test_dir, fileName))
