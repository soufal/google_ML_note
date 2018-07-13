#coding=utf-8
import tensorflow as tf 

#Create a graph
g = tf.Graph()

#将图创建为一个默认图
with g.as_default():
	#组装一个由以下三个操作组成的图表：
	#两个tf.constant 操作符来创建操作数
	#一个tf.add操作符来相加两个操作数
	x = tf.constant(8, name="x_const")
	y = tf.constant(5, name="y_const")
	sum = tf.add(x, y, name="x_y_sum")


	#创建一个会话
	#这个会话将会运行这个默认图。
	with tf.Session() as sess:
		print(sum.eval())