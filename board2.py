import tensorflow as tf
import numpy as np

# 导入数据
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 神经层里常见的参数通常有weights,biases和激励函数
def add_layer(
        inputs,
        in_size,
        out_size,
        n_layer,
        activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer ## define a new var
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            # tf.histogram_summary(layer_name+'/weights',Weights) # tensorflow 0.12以下版的
            tf.summary.histogram(layer_name + '/weights',Weights) # tensorflow >= 0.12

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1 , name='b')
            # tf.histogram_summary(layer_name+'/biase',biases)
            tf.summary.histogram(layer_name + '/biases' , biases) # tensorflow >= 0.12
        
        # 神经网络未激活的值，tf.matmul为矩阵乘法
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
        
        #当activation_function-激励函数为None时，输出就是当前的预测值-Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函         数中得到输出
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        # tf.histogram_summary(layer_name+'/outputs',outputs)
        tf.summary.histogram(layer_name + '/outputs', outputs) # tensorflow >= 0.12

    return outputs

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])

# 搭建网络 
# add hidden layer
l1 = add_layer(xs,1,10,n_layer = 1, activation_function = tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1,n_layer = 2, activation_function = None)

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
    # tf.scalar_summary('loss',loss) # tensorflow < 0.12
    tf.summary.scalar('loss',loss) # tensorflow >= 0.12

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# merged = tf.merge_all_summaries() # tensorflow < 0.12
merged = tf.summary.merge_all() # tensorflow >= 0.12

# writer = tf.train.SummaryWriter('logs/',sess.graph) # tensorflow < 0.12
writer = tf.summary.FileWriter("logs/",sess.graph) # tensorflow >= 0.12

#使用变量时，对要对它进行初始化，这是必不可少的
# sess.run(tf.initialize_all_variables()) # tf.initialize_all_variables() # tf 马上就要废弃这种写法
sess.run(tf.global_variables_initializer()) # 替换成这样就好

for i in range(1000):
    sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
    if i % 50 == 0:
        rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs,i)
