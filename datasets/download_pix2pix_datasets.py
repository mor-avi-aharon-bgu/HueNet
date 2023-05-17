import tensorflow as tf
import os

dataset_name = 'edges2shoes'  # cityscapes, facades, edges2shoes edges2handbags
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/' + dataset_name + '.tar.gz'

path_to_zip = tf.keras.utils.get_file(dataset_name + '.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), dataset_name + '/')
