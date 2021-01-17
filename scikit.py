import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# keep_prob保留概率 
keep_prob = tf.placeholder(tf.float32)
...
...
Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)

# X_train是训练数据，X_test是测试数据
digits = load_digits()
X = digits.data
Y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3)

# add output layer
l1 = add_layer(xs,64,50,'l1',activation_function = tf.nn.tanh)
prediction = add_layer(l1,50,10,'l2',activation_function = tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})

