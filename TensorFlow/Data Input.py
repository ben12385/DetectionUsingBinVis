import tensorflow as tf
import os

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_image_1024_malware(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.pad_to_bounding_box(image_decoded, 0, 0, 1024, 1024)
  image_complete_resized = tf.reshape(image_resized, [1, 1024, 1024, 3])
  
  image_converted = tf.image.convert_image_dtype(image_complete_resized, tf.float32)

  label = tf.reshape(label, [1, -1])
  return image_converted, label



# A vector of filenames.



safeFilePath = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\Clean\\'

labels = list()

listOfFilesSafe = sorted(os.listdir(safeFilePath))

#Add path extension to file name
for i in range(0, len(listOfFilesSafe)):
    listOfFilesSafe[i] = safeFilePath + listOfFilesSafe[i]
    labels.append(0)


maliciousFilePath = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\Malicious\\'
listOfFilesMalware = sorted(os.listdir(maliciousFilePath))

#Add path extension to file name
for i in range(0, len(listOfFilesMalware)):
    listOfFilesMalware[i] = maliciousFilePath + listOfFilesMalware[i]
    labels.append(1)


filenames = tf.constant(listOfFilesSafe + listOfFilesMalware);

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_image_1024_malware)

iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

def convLayer(input, shape, num):
    with tf.name_scope('conv' + str(num)):
        with tf.name_scope('weights'):
            W_conv = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            variable_summaries(W_conv)
        with tf.name_scope('biases'):
            b_conv = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
            variable_summaries(b_conv)
        with tf.name_scope('Wx_plus_b'):
            c_conv = tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding='SAME')
            h_conv = tf.nn.relu(c_conv + b_conv)
            tf.summary.histogram('pre_activations', h_conv)
        return h_conv


#---------------------------------------------CNN Structure----------------------------------------------------

h_conv1 = convLayer(next_element[0], [10, 10, 3, 2], 1)

h_conv2 = convLayer(h_conv1, [10, 10, 2, 5], 2)

h_conv3 = convLayer(h_conv2, [10, 10, 5, 10], 3)
   
with tf.name_scope('pool1'):
    h_pool1 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv4 = convLayer(h_pool1, [4, 4, 10, 15], 4)

h_conv5 = convLayer(h_conv4, [4, 4, 15, 20], 5)

h_conv6 = convLayer(h_conv5, [4, 4, 20, 25], 6)
   
with tf.name_scope('pool2'):
    h_pool2 = tf.nn.max_pool(h_conv6, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

h_conv7 = convLayer(h_pool2, [2, 2, 25, 26], 7)

h_conv8 = convLayer(h_conv7, [2, 2, 26, 27], 8)

h_conv9 = convLayer(h_conv8, [2, 2, 27, 28], 9)
   
with tf.name_scope('pool3'):
    h_pool3 = tf.nn.max_pool(h_conv9, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    print(h_pool3)

with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.truncated_normal([28672, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_conv1_flat = tf.reshape(h_pool3, [-1, 28672])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

with tf.name_scope('fc2'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 1], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]))
    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    tf.summary.scalar("test", next_element[1])

with tf.name_scope('loss'):
   cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=next_element[1], logits=y)
   cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 0), tf.argmax(next_element[1], 0))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('C:/Users/Ben/Desktop/Tensorflow_Test_Log' + '/update')
train_writer.add_graph(tf.get_default_graph())

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(iterator.initializer)
    print("Starting")
    for i in range(0,len(listOfFilesSafe)+len(listOfFilesMalware)):
        
        #print(train_step.run(feed_dict={x_image: next_element[0], y_: next_element[1]}))
        summary, output = sess.run([merged,train_step])
        train_writer.add_summary(summary, i)



print("Done")