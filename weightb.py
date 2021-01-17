import tensorflow as tf
python from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot = true)

def weight_variable(shape):
    inital = tf.truncted_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_poo_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1])

# 输入的placeholder
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

# dropout的placeholder,它是解决过拟合的有效手段
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])

# 第一层卷积
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2*2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2*2(h_conv2)

# 建立全连接层，通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据，-1表示先不考虑输入图片例子维度，将上一个输出结果展平
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_drop)

W_fc2 = weight_variable([1024,10]) 
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_dropt,W_fc2),b_fc2)

# 交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices = [1]))
# 优化器优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())





