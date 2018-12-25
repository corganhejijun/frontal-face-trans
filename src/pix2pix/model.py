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
    def __init__(self, sess, image_size=256, batch_size=1, sample_size=1, output_size=256, load_size=286,
                 gf_dim=64, df_dim=64, L1_lambda=100, input_c_dim=3, output_c_dim=3, dataset_name='facades',
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
        self.load_size = load_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.g_bn_e2_64 = batch_norm(name='g_bn_e2_64')
        self.g_bn_e3_64 = batch_norm(name='g_bn_e3_64')
        self.g_bn_e4_64 = batch_norm(name='g_bn_e4_64')
        self.g_bn_e5_64 = batch_norm(name='g_bn_e5_64')
        self.g_bn_e6_64 = batch_norm(name='g_bn_e6_64')
        self.g_bn_e7_64 = batch_norm(name='g_bn_e7_64')

        self.g_bn_d1_64 = batch_norm(name='g_bn_d1_64')
        self.g_bn_d2_64 = batch_norm(name='g_bn_d2_64')
        self.g_bn_d3_64 = batch_norm(name='g_bn_d3_64')
        self.g_bn_d4_64 = batch_norm(name='g_bn_d4_64')
        self.g_bn_d5_64 = batch_norm(name='g_bn_d5_64')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                [self.batch_size, self.image_size, self.image_size, self.input_c_dim + self.output_c_dim],
                name='real_A_and_B_images')

        self.real_B_256 = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B_64 = tf.image.resize_images(self.real_B_256, (int(self.image_size / 4), int(self.image_size / 4)))
        self.real_B_128 = tf.image.resize_images(self.real_B_256, (int(self.image_size / 2), int(self.image_size / 2)))
        self.real_A_256 = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.real_A_64 = tf.image.resize_images(self.real_A_256, (int(self.image_size / 4), int(self.image_size / 4)))
        self.real_A_128 = tf.image.resize_images(self.real_A_256, (int(self.image_size / 2), int(self.image_size / 2)))

        self.fake_B_64 = self.generator_128_to_64(self.real_A_128)
        self.fake_B_128 = self.generator_64_to_128(self.fake_B_64)
        self.fake_B_256 = self.generator_128_to_256(self.fake_B_128)

        self.real_AB_64 = tf.concat([self.real_A_64, self.real_B_64], 3)
        self.fake_AB_64 = tf.concat([self.real_A_64, self.fake_B_64], 3)
        self.BB_64 = tf.concat([self.real_B_64, self.fake_B_64], 3)
        self.BB_128 = tf.concat([self.real_B_128, self.fake_B_128], 3)
        self.BB_256 = tf.concat([self.real_B_256, self.fake_B_256], 3)
        self.D64, self.D_logits64 = self.discriminator(self.real_AB_64, "discriminator_64", reuse=False)
        self.D_64, self.D_logits_64 = self.discriminator(self.fake_AB_64, "discriminator_64", reuse=True)
        self.D64_BB, self.D_logits64_BB = self.discriminator(self.BB_64, "discriminator_64_BB")
        self.D128_BB, self.D_logits128_BB = self.discriminator(self.BB_128, "discriminator_128_BB", size=128)
        self.D256_BB, self.D_logits256_BB = self.discriminator(self.BB_256, "discriminator_256_BB", size=256)

        self.fake_B_sample = self.sampler(self.real_A_128)
        self.fake_B_sample_64 = self.sampler_64(self.real_A_128)
        self.fake_B_sample_128 = self.sampler_128(self.real_A_128)

        self.d64_sum = tf.summary.histogram("d64", self.D64)
        self.d_64_sum = tf.summary.histogram("d_64", self.D_64)
        self.d64_sum_bb = tf.summary.histogram("d64_bb", self.D64_BB)
        self.d128_sum_bb = tf.summary.histogram("d128_bb", self.D128_BB)
        self.d256_sum_bb = tf.summary.histogram("d256_bb", self.D256_BB)
        self.fake_B_64_sum = tf.summary.image("fake_B_64", self.fake_B_64)
        self.fake_B_128_sum = tf.summary.image("fake_B_128", self.fake_B_128)
        self.fake_B_256_sum = tf.summary.image("fake_B_256", self.fake_B_256)

        self.d_loss_real_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits64, labels=tf.ones_like(self.D64)))
        self.d_loss_fake_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_64, labels=tf.zeros_like(self.D_64)))
        self.d_loss_bb_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits64_BB, labels=tf.zeros_like(self.D64_BB)))
        self.d_loss_bb_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits128_BB, labels=tf.zeros_like(self.D128_BB)))
        self.d_loss_bb_256 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits256_BB, labels=tf.zeros_like(self.D256_BB)))
        self.g_loss_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_64, labels=tf.ones_like(self.D_64))) \
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits64_BB, labels=tf.ones_like(self.D64_BB))) \
                + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B_64 - self.fake_B_64))
        self.g_loss_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits128_BB, labels=tf.ones_like(self.D128_BB))) \
                + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B_128 - self.fake_B_128))
        self.g_loss_256 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits256_BB, labels=tf.ones_like(self.D256_BB))) \
                + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B_256 - self.fake_B_256))

        self.d_loss_real_64_sum = tf.summary.scalar("d_loss_real_64", self.d_loss_real_64)
        self.d_loss_fake_64_sum = tf.summary.scalar("d_loss_fake_64", self.d_loss_fake_64)
        self.d_loss_bb_64_sum = tf.summary.scalar("d_loss_bb_64", self.d_loss_bb_64)
        self.d_loss_bb_128_sum = tf.summary.scalar("d_loss_bb_128", self.d_loss_bb_128)
        self.d_loss_bb_256_sum = tf.summary.scalar("d_loss_bb_256", self.d_loss_bb_256)

        self.d_loss_64 = self.d_loss_real_64 + self.d_loss_fake_64 + self.d_loss_bb_64
        self.d_loss_128 = self.d_loss_bb_128
        self.d_loss_256 = self.d_loss_bb_256

        self.g_loss_64_sum = tf.summary.scalar("g_loss_64", self.g_loss_64)
        self.g_loss_128_sum = tf.summary.scalar("g_loss_128", self.g_loss_128)
        self.g_loss_256_sum = tf.summary.scalar("g_loss_256", self.g_loss_256)
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
        sample = [load_data(sample_file, self.image_size, self.load_size) for sample_file in data]
        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss_256, self.g_loss_256], feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        d_optim_64 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_loss_64, var_list=self.d_vars_64)
        g_optim_64 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss_64, var_list=self.g_vars_64)
        d_optim_128 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_loss_128, var_list=self.d_vars_128)
        g_optim_128 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss_128, var_list=self.g_vars_128)
        d_optim_256 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_loss_256, var_list=self.d_vars_256)
        g_optim_256 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss_256, var_list=self.g_vars_256)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum_64 = tf.summary.merge([self.d_64_sum, self.d64_sum_bb, self.fake_B_64_sum,
                self.d_loss_fake_64_sum, self.d_loss_bb_64_sum, self.g_loss_64_sum])
        self.d_sum_64 = tf.summary.merge([self.d64_sum, self.d_loss_real_64_sum, self.d_loss_64_sum])
        self.g_sum_128 = tf.summary.merge([self.d128_sum_bb, self.fake_B_128_sum, self.d_loss_bb_128_sum, self.g_loss_128_sum])
        self.d_sum_128 = tf.summary.merge([self.d_loss_128_sum])
        self.g_sum_256 = tf.summary.merge([self.d256_sum_bb, self.fake_B_256_sum, self.d_loss_bb_256_sum, self.g_loss_256_sum])
        self.d_sum_256 = tf.summary.merge([self.d_loss_256_sum])
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
            # only use train_size images from total data
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, self.image_size, self.load_size) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                for _ in range(2):
                    # Update D64 network
                    _, summary_str = self.sess.run([d_optim_64, self.d_sum_64], feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                for _ in range(4):
                    # Update G64 network
                    _, summary_str = self.sess.run([g_optim_64, self.g_sum_64], feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                for _ in range(1):
                    # Update D128 network
                    _, summary_str = self.sess.run([d_optim_128, self.d_sum_128], feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                for _ in range(2):
                    # Update G128 network
                    _, summary_str = self.sess.run([g_optim_128, self.g_sum_128], feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                for _ in range(1):
                    # Update D256 network
                    _, summary_str = self.sess.run([d_optim_256, self.d_sum_256], feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                for _ in range(2):
                    # Update G256 network
                    _, summary_str = self.sess.run([g_optim_256, self.g_sum_256], feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                errD_64 = self.d_loss_64.eval({self.real_data: batch_images})
                errD_128 = self.d_loss_128.eval({self.real_data: batch_images})
                errD_256 = self.d_loss_256.eval({self.real_data: batch_images})
                errG_64 = self.g_loss_64.eval({self.real_data: batch_images})
                errG_128 = self.g_loss_128.eval({self.real_data: batch_images})
                errG_256 = self.g_loss_256.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: [%.8f, %.8f, %.8f], g_loss: [%.8f, %.8f, %.8f]" \
                        % (epoch, idx, batch_idxs, time.time() - start_time, 
                                errD_64, errD_128, errD_256, errG_64, errG_128, errG_256))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 500) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, name, size=64, y=None, reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            # for        image->h0-> h1->h2->h3->h4
            #     256x256: 256->128->64->32->16->8
            #     128x128: 128-> 64->32->16->8
            #     64x64:    64-> 32->16->8
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            if size == 64:
                return tf.nn.sigmoid(h2), h2
            if size > 64:
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            if size == 128:
                return tf.nn.sigmoid(h3), h3
            h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, name='d_h4_conv')))
            return tf.nn.sigmoid(h4), h4
            
    def residual_block(self, image, size=128, type=1):
        rb1 = conv2d(image, self.gf_dim * 2, d_h=1, d_w=1, name='g_rb_' + str(size) + '_conv_rb' + str(type) + '_1')
        rb1 = tf.nn.relu(rb1)
        rb2 = conv2d(rb1, self.output_c_dim, d_h=1, d_w=1, name='g_rb_' + str(size) + '_conv_rb' + str(type) + '_2')
        rb_sum = tf.add(image, rb2, name='g_rb_' + str(size) + '_add_rb' + str(type))
        return tf.nn.relu(rb_sum)

    def g_64_to_128(self, image):
        s2 = int(self.output_size/2)
        rb1 = self.residual_block(image)
        rb2 = self.residual_block(rb1, type=2)
        rb2 = tf.image.resize_images(rb2, (s2, s2))
        self.d7  = conv2d(rb2, self.output_c_dim, d_h=1, d_w=1, name='g_d7_128')
        # d7 is (128 x 128 x self.gf_dim*1*2)
        return tf.nn.tanh(self.d7)

    def generator_64_to_128(self, image):
        with tf.variable_scope("generator"):
            return self.g_64_to_128(image)

    def g_128_to_256(self, image):
        s = self.output_size
        rb1 = self.residual_block(image, size=256, type=1)
        rb2 = self.residual_block(rb1, size=256, type=2)
        rb2 = tf.image.resize_images(rb2, (s, s))
        self.d8 = conv2d(rb2, self.output_c_dim, d_h=1, d_w=1, name='g_d8_256')
        # d8 is (256 x 256 x output_c_dim)
        return tf.nn.tanh(self.d8)

    def generator_128_to_256(self, image):
        with tf.variable_scope("generator"):
            return self.g_128_to_256(image)

    def g_128_to_64(self, image):
        s = self.output_size
        s4, s8, s16, s32, s64, s128 = int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (128 x 128 x input_c_dim)
        e1 = conv2d(image, self.gf_dim, name='g_e1_conv_64')
        # e1 is (64 x 64 x self.gf_dim)
        e2 = self.g_bn_e2_64(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv_64'))
        # e2 is (32 x 32 x self.gf_dim*2)
        e3 = self.g_bn_e3_64(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv_64'))
        # e3 is (16 x 16 x self.gf_dim*4)
        e4 = self.g_bn_e4_64(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv_64'))
        # e4 is (8 x 8 x self.gf_dim*8)
        e5 = self.g_bn_e5_64(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv_64'))
        # e5 is (4 x 4 x self.gf_dim*8)
        e6 = self.g_bn_e6_64(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv_64'))
        # e6 is (2 x 2 x self.gf_dim*8)
        e7 = self.g_bn_e7_64(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv_64'))
        # e7 is (1 x 1 x self.gf_dim*8)

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e7),
            [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1_64', with_w=True)
        d1 = tf.nn.dropout(self.g_bn_d1_64(self.d1), 0.5)
        d1 = tf.concat([d1, e6], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
            [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2_64', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2_64(self.d2), 0.5)
        d2 = tf.concat([d2, e5], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
            [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3_64', with_w=True)
        d3 = tf.nn.dropout(self.g_bn_d3_64(self.d3), 0.5)
        d3 = tf.concat([d3, e4], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
            [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4_64', with_w=True)
        d4 = self.g_bn_d4_64(self.d4)
        d4 = tf.concat([d4, e3], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
            [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5_64', with_w=True)
        d5 = self.g_bn_d5_64(self.d5)
        d5 = tf.concat([d5, e2], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
            [self.batch_size, s4, s4, self.output_c_dim], name='g_d6_64', with_w=True)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        return tf.nn.tanh(self.d6)

    def generator_128_to_64(self, image):
        with tf.variable_scope("generator"):
            return self.g_128_to_64(image)

    def sampler_64(self, image):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            return self.g_128_to_64(image)

    def sampler_128(self, image, y=None):
        out_64 = self.sampler_64(image)

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            return self.g_64_to_128(out_64)
            
    def sampler(self, image, y=None):
        out_128 = self.sampler_128(image)

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            return self.g_128_to_256(out_128)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

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
            sample = [load_data(sample_file, args.test_size, self.load_size, is_test=True) for sample_file in sample_files]

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
                samples = self.sess.run(self.fake_B_sample, feed_dict={self.real_data: sample_image})
                save_images(samples, [self.batch_size, 1], './{}/{}.png'.format(args.test_dir, fileName))

                samples = self.sess.run(self.fake_B_sample_64, feed_dict={self.real_data: sample_image})
                save_images(samples, [self.batch_size, 1], './{}_64/{}.png'.format(args.test_dir, fileName))

                samples = self.sess.run(self.fake_B_sample_128, feed_dict={self.real_data: sample_image})
                save_images(samples, [self.batch_size, 1], './{}_128/{}.png'.format(args.test_dir, fileName))
