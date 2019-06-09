import tensorflow as tf
import numpy as np
import cv2

sess = tf.Session()
saver = tf.train.import_meta_graph('models/model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('models'))
inputs = sess.graph.get_tensor_by_name('input:0')
pred_cls_p = sess.graph.get_tensor_by_name('pred_cls:0')
pred_loc_p = sess.graph.get_tensor_by_name('pred_loc:0')

image = cv2.imread('51.png')
image = cv2.resize(image, (320, 240)) / 255.
image = np.expand_dims(image, 0)

import time

start = time.time()
pred_cls,pred_loc = sess.run([pred_cls_p,pred_loc_p], feed_dict={inputs: image})
print(np.sum(pred_cls[...,1:]))
end2 = time.time()
