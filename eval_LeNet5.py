import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference_LeNet5
import Train_LeNet5
import numpy as np
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # x = tf.placeholder(tf.float32, [
        #     Train_LeNet5.BATCH_SIZE,
        #     inference_LeNet5.IMAGE_SIZE,
        #     inference_LeNet5.IMAGE_SIZE,
        #     inference_LeNet5.NUM_CHANNELS],
        #                    name='x-input')
        x = tf.placeholder(tf.float32, [
            Train_LeNet5.BATCH_SIZE,
            inference_LeNet5.IMAGE_SIZE,
            inference_LeNet5.IMAGE_SIZE,
            inference_LeNet5.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None,inference_LeNet5.OUTPUT_NODE], name='y-input')
        # validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 正则化
        regularizer = tf.contrib.layers.l2_regularizer(Train_LeNet5.REGULARIZATION_RATE)

        y = inference_LeNet5.inference(x, None,regularizer)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(Train_LeNet5.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(Train_LeNet5.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    xs, ys = mnist.validation.next_batch(Train_LeNet5.BATCH_SIZE)
                    # 重构测试集的shape
                    reshaped_xs = np.reshape(xs, (
                        Train_LeNet5.BATCH_SIZE,
                        inference_LeNet5.IMAGE_SIZE,
                        inference_LeNet5.IMAGE_SIZE,
                        inference_LeNet5.NUM_CHANNELS))
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: ys})
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()