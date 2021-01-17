import tensorflow as tf

# 在Tensorflow中需要定义placeholder的type，一般为float32形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply是将input1和input2做乘法运算，并输出为output
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict = {input1:[7.],input2:[2.]}))
# [14.]
