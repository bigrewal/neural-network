from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# W1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="w1")
# b1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.random_normal([784, 10], stddev=0.35),name="w2")
b2 = tf.Variable(tf.zeros([10]))

def my_model(x):
  # y1 = tf.nn.softmax(tf.matmul(x, W1)+b1)
  y = tf.nn.softmax(tf.matmul(x,W2)+b2)

  return y


pred = my_model(x)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_)
train_step = tf.train.AdamOptimizer().minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


print("Accuracy on the test set..")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))