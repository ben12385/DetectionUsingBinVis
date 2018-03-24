from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  path = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\test\\'
  filename = os.path.join(path, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(labels[index])),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())




def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels = 3, dtype = tf.uint8)
  image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 1024, 1024)
  image_casted = tf.image.convert_image_dtype(image_resized, dtype=tf.float32)
  return image_casted, 1


def main(unused_argv):

    #Input data from folder
    path = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\test\\'
    training_filenames = sorted(os.listdir(path))
    for a in range(0,len(training_filenames)):
        training_filenames[a] = path+training_filenames[a]


    # A vector of filenames.
    filenames = tf.constant(training_filenames)

    data_sets = tf.data.Dataset.from_tensor_slices(filenames)
    data_sets = data_sets.map(_parse_function)

    # Convert to Examples and write the result to TFRecords.
    convert_to(data_sets, 'train')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
