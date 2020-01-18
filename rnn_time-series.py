import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('petr4.csv')
data = data.dropna()

data = data.iloc[:,1].values

plt.plot(data)

periods = 30
future_prediction = 1

X = data[0:(len(data) - (len(data) % periods))]
X_batches = X.reshape(-1, periods, 1)

y = data[1:(len(data) - (len(data) % periods)) + future_prediction]
y_batches = y.reshape(-1, periods, 1)

X_test = data[-(periods + future_prediction):]
X_test = X_test[:periods]
X_test = X_test.reshape(-1, periods, 1)
y_test = data[-(periods):]
y_test = y_test.reshape(-1, periods, 1)

tf.reset_default_graph()

inputs = 1
hidden_neurons = 100
output_neurons = 1

xph = tf.placeholder(tf.float32, [None, periods, inputs])
yph = tf.placeholder(tf.float32, [None, periods, output_neurons])

#cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_neurons, activation = tf.nn.relu)
#cell = tf.contrib.rnn.LSTMCell(num_units = hidden_neurons, activation = tf.nn.relu)
#cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = 1)

def create_cell():
    return tf.contrib.rnn.LSTMCell(num_units = hidden_neurons, activation = tf.nn.relu)

def create_multiple_cells():
    cells =  tf.nn.rnn_cell.MultiRNNCell([create_cell() for i in range(4)])
    return tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob = 0.1)

cell = create_multiple_cells()

cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = 1)


rnn_output, _ = tf.nn.dynamic_rnn(cell, xph, dtype = tf.float32)
error = tf.losses.mean_squared_error(labels = yph, predictions = rnn_output)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training = optimizer.minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        _, cost = sess.run([training, error], feed_dict = {xph: X_batches, yph: y_batches})
        if epoch % 100 == 0:
            print(epoch + 1, ' error: ', cost)

    predictions = sess.run(rnn_output, feed_dict = {xph: X_test})


y_test.shape
y_test2 = np.ravel(y_test)

predictions2 = np.ravel(predictions)

mae = mean_absolute_error(y_test2, predictions2)

plt.plot(y_test2, '*', markersize = 10, label = 'Real Value')
plt.plot(predictions2, 'o', label = 'Predictions')
plt.legend()

plt.plot(y_test2, label = 'Real Value')
plt.plot(y_test2, 'w*', markersize = 10, color = 'red')
plt.plot(predictions2, label = 'Predictions')
plt.legend()
