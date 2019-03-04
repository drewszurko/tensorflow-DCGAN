from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import DCGAN

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size of images to train.')
flags.DEFINE_integer('width', 64, 'Image input width.')
flags.DEFINE_integer('height', 64, 'Image input height.')
flags.DEFINE_string('dataset', 'cifar10',
                    'Dataset to train [cifar10, celeb_a, tf_flowers]')
flags.DEFINE_string('cache', '',
                    'Optional: [None, memory, disk]. If specified, data will '
                    'be cached for faster training.\nmemory: slower, '
                    'disposable.\ndisk: faster, requires space.')
flags.DEFINE_boolean('crop', False, 'Center crop image.')
flags.DEFINE_string('output_dir', '', 'Directory to save output.')
FLAGS = flags.FLAGS


def main(_):
    dcgan = DCGAN(height=FLAGS.height, width=FLAGS.width,
                  batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
                  learning_rate=FLAGS.learning_rate)
    dcgan.train()


if __name__ == '__main__':
    tf.app.run()
