from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

import tensorflow as tf
import numpy as np

# 搭建网络
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1 , name ='b')
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(vxs,vys):
    global prediction
    y_pre = sess.run(prediction,feed_dict = {xs:vxs})
    err = tf.equal(tf.argmax(y_pre,1),tf.argmax(vys,1))
    acc = tf.reduce_mean(tf.cast(err,tf.float32))
    result = sess.run(acc,feed_dict = {xs:vxs,ys:vys})
    return result

# 调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层
prediction = add_layer(xs,784,10,activation_function = tf.nn.softmax)

# loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices = [1])) # loss

# train方法（最优化算法）采用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
# tf.initialize_all_variables() 这种写法马上就要被废弃
# 替换成下面的写法：
sess.run(tf.global_variables_initializer())

# 现在开始train,每次只取100张图片，免得数据太多训练太慢
batch_xs,batch_ys = mnist.train.next_batch(100)
sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys})

# 每训练50次输出一下预测精度
for i in range(1000):
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

