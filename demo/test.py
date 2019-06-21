import tinyflow as tf
import numpy as np

x = tf.Placeholder()
k = tf.Variable(init_value=5)
b = tf.Variable(init_value=3)
y = tf.add(tf.multiply(k, x), b) # y = k * x + b

X = np.linspace(-3, 10, 1000)
Y = 2 * X + 7

lr = 0.01
for _ in range(2000):
    i = np.random.randint(1000)
    data_x, data_y = X[i], Y[i]
    x.feed(data_x)
    loss = y.eval() - data_y
    y.backward(loss * lr)
    print(k.eval(), b.eval())
