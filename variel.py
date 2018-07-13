#coding=utf-8
import tensorflow as tf

g = tf.Graph()
with g.as_default():
  # 创建一个初始值为3的变量。
  v = tf.Variable([3])

  # 创建一个形状为[1]，有着随机初始值的变量,
  # 从正态分布中取样，平均值为1，标准差为0.35
  w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
  with tf.Session() as sess:
  	initialization = tf.global_variables_initializer()
  	sess.run(initialization)
  	#这样，变量就能够被正常的接受，并分配值给他们。
  	print(v.eval())
  	print(w.eval())
  	#改变变量的值
  	assignment = tf.assign(v, [7])
  	#变量还没有被改变
  	print(v.eval())
  	#需要执行赋值指令
  	sess.run(assignment)
  	#现在变量才改变
  	print(v.eval())
