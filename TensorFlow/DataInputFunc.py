import tensorflow as tf
import os

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_image_1024_malware(filename, labels):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.pad_to_bounding_box(image_decoded, 0, 0, 1024, 1024)

  #image_complete_resized = tf.reshape(image_resized, [1024, 1024, 3])
  
  image_converted = tf.image.convert_image_dtype(image_resized, tf.float32)

  return image_converted, labels

def prepDataSet(path):
    filenames = sorted(os.listdir(path))
    for i in range(0, len(filenames)):
        filenames[i] = path+filenames[i]

    filenames_tensor = tf.constant(filenames)
    labels_tensor = tf.constant([1])

    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
    dataset = dataset.map(_parse_image_1024_malware)
    #dataset = dataset.repeat()
    #dataset.shuffle(buffer_size=10000)

    return dataset.batch(1)