#coding=utf-8
import tensorflow as tf
#task2：假设模拟投掷两个骰子 10 次。
#创建一个骰子模拟，在模拟中生成一个 10x3 二维张量，其中：
#列 1 和 2 均存储一个骰子的一次投掷值。
#列 3 存储同一行中列 1 和 2 的值的总和。
#例如，第一行中可能会包含以下值：
#列 1 存储 4
#列 2 存储 3
#列 3 存储 7
with tf.Graph().as_default(), tf.Session() as sess:
  dics1 = tf.Variable(tf.random_uniform([10,1], minval=1, maxval=7,
                                                  dtype=tf.int32))
  dics2 = tf.Variable(tf.random_uniform([10,1], minval=1, maxval=7,
                                                  dtype=tf.int32))
  dics_sum = tf.add(dics1, dics2)
  
  resulting_matrix = tf.concat(
      values=[dics1, dics2, dics_sum], axis=1)
  sess.run(tf.global_variables_initializer())
  print(resulting_matrix.eval())