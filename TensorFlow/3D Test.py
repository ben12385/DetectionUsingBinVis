import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding = 'SAME')

#Generate 3D Model
with open("test.7z", "rb") as f:
    byteString = f.read()

batch = 0;
height = 256;
width = 256;
depth = 256;

img = np.zeros((1, height, width, depth, 1), np.float32)

     
iterations = (int)(len(byteString)/3)
for x in range(0, iterations):
    currentRow = bytearray(byteString[x*3])[0]
    currentColumn = bytearray(byteString[x*3+1])[0]
    currentDepth = bytearray(byteString[x*3+2])[0]
    img[batch, currentRow, currentColumn, currentDepth] = img[batch, currentRow, currentColumn, currentDepth] + 1



print("Image Created, entering NN")
#CNN Model
#x_cube = tf.reshape(x, [-1, 256, 256, 256, 1])

x = tf.placeholder(tf.float32, [1, 256, 256, 256, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope("First_3D_Conv_Layer"):
    w_conv1 = weight_variable([10, 10, 10, 1, 2])
    b_conv1 = bias_variable([2])

    h_conv1 = tf.nn.relu(conv3d(x, w_conv1) + b_conv1)


with tf.name_scope("First_Pooling_Layer"):
    h_pool1 = tf.nn.max_pool3d(h_conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

with tf.name_scope("First_Deeply_Connected_Layer"):
    flatten_h_conv1 = tf.reshape(h_pool1, [1, -1])
    W_fc1 = weight_variable([128*128*128*2, 1])
    b_fc1 = bias_variable([1])

    y_ = tf.matmul(flatten_h_conv1, W_fc1) + b_fc1

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=[1], logits= y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(y_, 1)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Start Training")
    output = train_step.run(feed_dict={x: img})
    print(output)

    #train_accuracy = accuracy.eval(feed_dict={x: img})
    #print('training accuracy %g' % (train_accuracy))

print("Completed")