import tensorflow as tf
import os

#Placeholder for input and output
x_image = tf.placeholder(tf.float32, shape=[None, 1024, 1024, 3])

with tf.name_scope('conv1'):
    W_conv1 = tf.Variable(tf.truncated_normal([10, 10, 3, 5], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[5]))
    c_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(c_conv1 + b_conv1)
    
with tf.name_scope('pool1'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME')

with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.truncated_normal([327680, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_conv1_flat = tf.reshape(h_conv1, [-1, 327680])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

with tf.name_scope('fc2'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 1], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]))
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=[1], logits=y)
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)














path = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\test\\'
filenames = sorted(os.listdir(path))
for a in range(0, len(filenames)):
    filenames[a] = path + filenames[a]

training_filenames = tf.constant(filenames)

filename_queue = tf.train.string_input_producer(training_filenames)

image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_png(image_file)
image_resize = tf.image.resize_image_with_crop_or_pad(image, 1024, 1024)
image_cast = tf.image.convert_image_dtype(image_resize, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    train_step.run(feed_dict={x_image: [image_cast]})

coord.request_stop()
coord.join(threads)

print("Done")