import tensorflow as tf

con = tf.Constant(value=[1, 2])
with tf.Session() as sess:
    print(sess.run(con))