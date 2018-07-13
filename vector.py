#coding=utf-8
import tensorflow as tf 

with tf.Graph().as_default():
	#创建一个由6个元素的向量（1-D张量）。
	primes = tf.constant([2, 3, 4, 7, 11, 13], dtype=tf.int32)

	#创建另一个有6个元素的向量，每一个元素在向量中将被初始化为1.
	#第一个参数[6]是装量的形状，表示6个元素。
	#可以使用广播机制，设置形状的大小只有小于primes的大小即可，比如1,2,3,4,5
	ones = tf.ones([6], dtype=tf.int32)

	#相加两个向量，结果张量是一个6个元素的向量。
	just_beyond_primes = tf.add(primes, ones)

	#创建一个会话去运行这个默认的图。
	with tf.Session() as sess:
		print(just_beyond_primes.eval())

	#一个标量（0-D张量）
	scalar = tf.zeros([])

	#一个3个元素的向量
	vector = tf.zeros([3])

	#一个2行3列的矩阵
	matrix = tf.zeros([2, 3])

	with tf.Session() as sess:
		print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval())
		print('vector has shape', vector.get_shape(), 'and value:\n', vector.eval())
		print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval())

	#矩阵乘法
	#创建一个3行4列的矩阵（2-d张量）
	x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype=tf.int32)

	#创建一个4行3列的矩阵
	y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

	#x和y相乘
	#得到一个3行2列的矩阵
	matrix_multiply_result = tf.matmul(x, y)
	with tf.Session() as sess:
		print(matrix_multiply_result.eval())

	#张量变形
	# Create an 8x2 matrix (2-D tensor).
	matrix = tf.constant([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16]], dtype=tf.int32)
	# Reshape the 8x2 matrix into a 2x8 matrix.
	reshaped_2x8_matrix = tf.reshape(matrix, [2,8])
	# Reshape the 8x2 matrix into a 4x4 matrix
	reshaped_4x4_matrix = tf.reshape(matrix, [4,4])

	with tf.Session() as sess:
		print("Original matrix (8x2):")
		print(matrix.eval())
		print("Reshaped matrix (2x8):")
		print(reshaped_2x8_matrix.eval())
		print("Reshaped matrix (4x4):")
		print(reshaped_4x4_matrix.eval())

	with tf.Graph().as_default():
	# Create an 8x2 matrix (2-D tensor).
	matrix = tf.constant([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16]], dtype=tf.int32)
	# Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  	reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])
  	# Reshape the 8x2 matrix into a 1-D 16-element tensor.
  	one_dimensional_vector = tf.reshape(matrix, [16])
  	with tf.Session() as sess:
  		print("Original matrix (8x2):")
  		print(matrix.eval())
  		print("Reshaped 3-D tensor (2x2x4):")
  		print(reshaped_2x2x4_tensor.eval())
  		print("1-D vector:")
  		print(one_dimensional_vector.eval())