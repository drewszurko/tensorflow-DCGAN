from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

FLAGS = tf.flags.FLAGS
TMP_DIR = Path(tempfile.gettempdir())
CACHE_DIR = TMP_DIR.joinpath('cache')


def load_dataset():
    """Load dataset for training.

    Returns:
        dataset: Dataset to train on.
        total_samples: Total number of samples in dataset.
    """
    dataset_builder = tfds.builder(FLAGS.dataset)
    dataset_builder.download_and_prepare()
    total_samples = dataset_builder.info.splits.total_num_examples
    dataset = dataset_builder.as_dataset(split=tfds.Split.ALL)
    dataset = dataset.shuffle(buffer_size=15000, reshuffle_each_iteration=True)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        preprocess_image, FLAGS.batch_size, drop_remainder=True,
        num_parallel_calls=tf.data.experimental.AUTOTUNE))
    dataset = dataset_cache(dataset)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, total_samples


def preprocess_image(dataset):
    """Preprocess dataset images.

    Args:
        dataset: Dataset to train.

    Returns:
        Normalized dataset images for training.
    """
    image = dataset['image']
    if FLAGS.crop:
        image = tf.image.central_crop(image, 0.5)
    image = tf.image.resize_images(image, [FLAGS.width, FLAGS.height])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def dataset_cache(dataset):
    """Set dataset cache.

    Args:
        dataset: Dataset to train.

    Returns:
        Dataset with optional user specified cache.
    """
    if not FLAGS.cache:
        return dataset
    elif FLAGS.cache == 'memory':
        return dataset.cache()
    elif FLAGS.cache == 'disk':
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for p in CACHE_DIR.glob(FLAGS.dataset + '*'):
            p.unlink()
        return dataset.cache(str(CACHE_DIR / FLAGS.dataset))
    else:
        raise RuntimeError(
            '{0} is not a valid cache option [None, memory, disk]'.format(
                FLAGS.dataset))


def merge_images(images, size=(8, 8)):
    """Merge generated image arrays into a single image array.

    Args:
        images: Batch of predicted image arrays generated from noise.
        size: (H, W) number of images to merge.

    Returns:
        Merged image array.
    """
    height, width = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        chans = images.shape[3]
        img = np.zeros((height * size[0], width * size[1], chans))
        for idx, image in enumerate(images):
            i = idx % size[1]
            d = idx // size[1]
            img[d * height:d * height + height, i * width:i * width + width,
            :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((height * size[0], width * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            d = idx // size[1]
            img[d * height:d * height + height,
            i * width:i * width + width] = image[:, :, 0]
        return img
    else:
        raise ValueError('merge_images(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def save_generated_images(generated_images, epoch):
    """Save generated images.

    Args:
        generated_images: Batch of predicted images generated from noise.
        epoch: Epoch belonging to batch of generated images.
    """
    image = (generated_images + 1) * 127.5
    image = merge_images(image)
    i = Image.fromarray(image.astype(np.uint8), mode='RGB')
    save_name = FLAGS.dataset + '_{}.png'.format(epoch)
    save_dir = os.path.join(FLAGS.output_dir, save_name)
    i.save(save_dir)
