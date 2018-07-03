import tensorflow as tf

input1 = tf.constant([3.0,2.0,3.0],name='input-1')
input2 = tf.Variable(tf.random_uniform([3]),name='input-2')
output = tf.add_n([input1,input2],name='add')

writer = tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
writer.close()
# import tensorflow as tf;
#
# # 定义一个简单的计算图，实现向量加法操作
# input1 = tf.constant([1.0, 2.0, 3.0], name="input1");
# input2 = tf.Variable(tf.random_uniform([3], name="input2"));
# output = tf.add_n([input1, input2], name="add");
#
# # 生产一个写日志的writer，并将当前的TensorFlow计算图写入日志，TensorFlow提供了多种写日志的API
# writer = tf.summary.FileWriter("c:/python35/tensorlog/show01", tf.get_default_graph())
# writer.close()