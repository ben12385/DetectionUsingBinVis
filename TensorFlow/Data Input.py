import tensorflow as tf
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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



safeFilePath = './Images/'

labels = list()

listOfFilesSafe = sorted(os.listdir(safeFilePath))

#Add path extension to file name
for i in range(0, len(listOfFilesSafe)):
    listOfFilesSafe[i] = safeFilePath + listOfFilesSafe[i]
    labels.append([0,1])


maliciousFilePath = './MalwareImages/'
listOfFilesMalware = sorted(os.listdir(maliciousFilePath))

#Add path extension to file name
for i in range(0, len(listOfFilesMalware)):
    listOfFilesMalware[i] = maliciousFilePath + listOfFilesMalware[i]
    labels.append([1,0])


total = listOfFilesSafe + listOfFilesMalware
random.shuffle(total)
	
filenames = tf.constant(total)

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


def poolLayer(input, shape, num):
    with tf.name_scope('pool' + str(num)):
        h_pool = tf.nn.max_pool(input, shape, shape, padding='SAME' )
    return h_pool

#---------------------------------------------CNN Structure----------------------------------------------------

h_conv1 = convLayer(next_element[0], [4, 4, 3, 8], 1)

h_conv2 = convLayer(h_conv1, [4, 4, 8, 8], 2)

h_pool1 = poolLayer(h_conv2, [1, 2, 2, 1], 1)

h_conv3 = convLayer(h_pool1, [4, 4, 8, 12], 3)

h_conv4 = convLayer(h_conv3, [4, 4, 12, 12], 3)
   
h_pool2 = poolLayer(h_conv4, [1, 2, 2, 1], 2)

h_conv5 = convLayer(h_pool2, [4, 4, 12, 14], 3)

h_conv6 = convLayer(h_conv5, [4, 4, 14, 14], 3)
   
h_pool3 = poolLayer(h_conv6, [1, 2, 2, 1], 3)

h_conv7 = convLayer(h_pool3, [3, 3, 14, 16], 3)

h_conv8 = convLayer(h_conv7, [3, 3, 16, 16], 3)
   
h_pool4 = poolLayer(h_conv8, [1, 2, 2, 1], 3)

h_conv9 = convLayer(h_pool4, [2, 2, 16, 18], 3)

h_conv10 = convLayer(h_conv9, [2, 2, 18, 18], 3)
   
h_pool5 = poolLayer(h_conv10, [1, 2, 2, 1], 3)


with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.truncated_normal([18432, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_conv1_flat = tf.reshape(h_pool5, [-1, 18432])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

with tf.name_scope('fc2'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]))
    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    tf.summary.scalar('output', tf.reduce_mean(tf.argmax(y)))
    tf.summary.scalar('correct', tf.reduce_mean(tf.argmax(next_element[1])))

with tf.name_scope('loss'):
   loss = tf.losses.softmax_cross_entropy(onehot_labels=next_element[1], logits=y)
   tf.summary.scalar('loss', loss)
   

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    prediction_difference = tf.abs(tf.subtract(y, tf.to_float(next_element[1])))


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./ImageData')
train_writer.add_graph(tf.get_default_graph())

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2


saver = tf.train.Saver();

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print("Starting")
    for i in range(0,len(listOfFilesSafe)+len(listOfFilesMalware)/2):
        try:
            if(i%100 == 0):
                summary, output = sess.run([merged,train_step])
                train_writer.add_summary(summary, i)
		save_path = saver.save(sess, './CheckPoint/ImageModel.ckpt')
		print("Done" + str(i))
            else:
                summary, output = sess.run([merged,train_step])        
        except Exception as e:
            print(e)
print("Done")
