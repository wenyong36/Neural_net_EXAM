import numpy as np
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])


# 参数权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 可以将x乘以一个置换矩阵
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二次卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三次卷积
W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# 全连接层
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练与评估
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

# 读取矩阵
A = np.loadtxt('A.txt', dtype='int', delimiter=', ')
np.mat(A)

for i in range(10001):
    batch = mnist.train.next_batch(50)
    batch_permutation = (np.mat(batch[0]) * A).getA()   # 做置换
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_permutation, y_: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("step %d, training accuracy %f%%" % (i, train_accuracy*100))
    train_step.run(feed_dict={x: batch_permutation, y_: batch[1], keep_prob: 0.5})

test_permutation = (np.mat(mnist.test.images) * A).getA()   # 做置换
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: test_permutation, y_: mnist.test.labels, keep_prob: 1.0}))
