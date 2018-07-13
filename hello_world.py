import tensorflow as tf

c = tf.constant('Hello, world!')			#定义一个常量
with tf.Session() as sess:				#定义一个会话，图必须在会话中运行。
    print(sess.run(c))