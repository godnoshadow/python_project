import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data , y_data)
plt.ion()
plt.show()

for i in range(1000):
    # training 
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        # plot the prediction
        lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
        plt.pause(0.1)




