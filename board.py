import tensorflow as tf

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

xs = tf.placeholder(tf.float32,[None,1],name = 'x_in')
ys = tf.placeholder(tf.float32,[None,1],name = 'y_in')

with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])

def add_layer(inputs,in_size,out_size,activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        # define weights name
        with tf.name_scope('weights'):
             Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        # define biase
        with tf.name_scope('biases'):
             biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
        #define Wx_plus_b
        with tf.name_scope('Wx_plus_b'):
             Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b,)
        return outputs

sess = tf.Session() # get session
# tf.train.SummaryWriter soon be deprecated,use following
writer = tf.summary.FileWriter("logs/",sess.graph)
