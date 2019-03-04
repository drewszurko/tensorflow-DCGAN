from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras as k
from tqdm.autonotebook import tqdm

from utils import save_generated_images, load_dataset


class DCGAN:
    def __init__(self, epochs=100, learning_rate=0.0002, batch_size=64,
                 height=64, width=64):
        """Build DCGAN model.

        Args:
            epochs: (optional) Number of epochs to train model.
            learning_rate: (optional) Generator and Discriminator learning rate.
            batch_size: (optional) Samples to propagate through network.
            height: (optional) I/O image height.
            width: (optional) I/O image width.
         """

        seed = tf.constant(19, dtype=tf.int32, name='train_seed')
        self.init = tf.truncated_normal_initializer(stddev=0.02)
        self.height = height
        self.width = width
        self._z_dim = 100
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._beta1 = 0.5
        self.epochs = epochs
        self._dataset, self._total_images = load_dataset()
        self._chans = 3
        self._pred_noise = tf.random_normal([self.batch_size, self._z_dim],
                                            seed=seed)
        self._tfe = tf.contrib.eager
        self._gen_opt = tf.train.AdamOptimizer(self.learning_rate, self._beta1)
        self._disc_opt = tf.train.AdamOptimizer(self.learning_rate, self._beta1)
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def train(self):
        train_d_loss = self._tfe.metrics.Mean()
        train_g_loss = self._tfe.metrics.Mean()

        for epoch in range(self.epochs):
            epoch_d_loss = self._tfe.metrics.Mean()
            epoch_g_loss = self._tfe.metrics.Mean()

            with tqdm(total=self._total_images,
                      desc=('Epoch {0}/{1}'.format(epoch + 1, self.epochs)),
                      ncols=100) as pbar:
                for images in self._dataset:
                    g_loss, d_loss = self._train_step(images)
                    epoch_g_loss(g_loss)
                    epoch_d_loss(d_loss)
                    pbar.update(self.batch_size)

                train_d_loss(epoch_d_loss.result())
                train_g_loss(epoch_g_loss.result())

                pbar.set_postfix_str(
                    'Gen. Loss: {0:.4f} Disc. Loss: {1:.4f} Avg. Gen. Loss: '
                    '{2:.4f} Avg. Disc. Loss: {3:.4f}'.format(
                        epoch_g_loss.result(), epoch_d_loss.result(),
                        train_g_loss.result(), train_d_loss.result(),
                    )
                )

                predictions = self.predict(self._pred_noise)
                save_generated_images(predictions, epoch + 1)

    def _gen_loss(self, fake_logits):
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_logits),
                                               fake_logits)

    def _disc_loss(self, real_logits, fake_logits):
        real_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real_logits), logits=real_logits,
            label_smoothing=0.2
        )

        fake_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(fake_logits), logits=fake_logits,
            label_smoothing=0.2,
        )

        total_loss = tf.add(real_loss, fake_loss)
        return total_loss

    def predict(self, x):
        return self.generator(x, training=False)

    @tf.contrib.eager.defun
    def _train_step(self, images):
        noise = tf.random_normal([self.batch_size, self._z_dim])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)
            real_logits = self.discriminator(images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            g_loss = self._gen_loss(fake_logits)
            d_loss = self._disc_loss(real_logits, fake_logits)

        g_gradients = g_tape.gradient(g_loss, self.generator.variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.variables)

        self._gen_opt.apply_gradients(
            zip(g_gradients, self.generator.variables))
        self._disc_opt.apply_gradients(
            zip(d_gradients, self.discriminator.variables))

        return g_loss, d_loss

    def _build_generator(self):
        dim = self.height
        dim_shape = dim // 16

        model = k.Sequential([
            k.layers.Dense(dim * 8 * dim_shape * dim_shape,
                           kernel_initializer=self.init,
                           input_shape=(self._z_dim,), name='g_1_dense',
                           use_bias=False),
            k.layers.BatchNormalization(momentum=0.9, name='g_1_batch'),
            k.layers.ReLU(name='g_1_relu'),
            k.layers.Reshape((dim_shape, dim_shape, dim * 8),
                             name='g_1_reshape'),

            k.layers.Conv2DTranspose(dim * 4, 5, 2, 'same',
                                     kernel_initializer=self.init,
                                     use_bias=False, name='g_2_conv'),
            k.layers.BatchNormalization(momentum=0.9, name='g_2_batch'),
            k.layers.ReLU(name='g_2_relu'),

            k.layers.Conv2DTranspose(dim * 2, 5, 2, 'same',
                                     kernel_initializer=self.init,
                                     name='g_3_conv', use_bias=False),
            k.layers.BatchNormalization(momentum=0.9, name='g_3_batch'),
            k.layers.ReLU(name='g_3_relu'),

            k.layers.Conv2DTranspose(dim, 5, 2, 'same',
                                     kernel_initializer=self.init,
                                     name='g_4_conv', use_bias=False),
            k.layers.BatchNormalization(momentum=0.9, name='g_4_batch'),
            k.layers.ReLU(name='g_4_relu'),

            k.layers.Conv2DTranspose(self._chans, 5, 2, 'same',
                                     kernel_initializer=self.init,
                                     name='g_5_conv', use_bias=False),
            k.layers.Activation(activation='tanh', name='g_5_act')],
            name='Generator')

        model.summary()
        return model

    def _build_discriminator(self):
        dim = self.height

        model = k.models.Sequential([
            k.layers.Conv2D(dim, 5, 2, 'same', kernel_initializer=self.init,
                            input_shape=(self.height, self.width, self._chans),
                            use_bias=False, name='d_1_conv'),
            k.layers.BatchNormalization(momentum=0.9, name='d_1_bn'),
            k.layers.LeakyReLU(alpha=0.2, name='d_1_lr'),

            k.layers.Conv2D(dim * 2, 5, 2, 'same',
                            kernel_initializer=self.init,
                            name='d_2_conv', use_bias=False),
            k.layers.BatchNormalization(momentum=0.9, name='d_2_bn'),
            k.layers.LeakyReLU(alpha=0.2, name='d_2_lr'),

            k.layers.Conv2D(dim * 4, 5, 2, 'same',
                            kernel_initializer=self.init,
                            name='d_3_conv', use_bias=False),
            k.layers.BatchNormalization(momentum=0.9, name='d_3_bn'),
            k.layers.LeakyReLU(alpha=0.2, name='d_3_lr'),

            k.layers.Conv2D(dim * 8, 5, 2, 'same',
                            kernel_initializer=self.init,
                            name='d_4_conv', use_bias=False),
            k.layers.BatchNormalization(momentum=0.9, name='d_4_bn'),
            k.layers.LeakyReLU(alpha=0.2, name='d_4_lr'),

            k.layers.Flatten(name='d_5_flat'),
            k.layers.Dense(1, name='d_5_dense', kernel_initializer=self.init,
                           use_bias=False)], name='Discriminator')

        model.summary()
        return model
