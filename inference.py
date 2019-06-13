import tensorflow as tf
import numpy as np
import cv2
import time
from create_tfrecord import get_anchors,getColor

anchors = get_anchors()

sess = tf.Session()
saver = tf.train.import_meta_graph('models/hand-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('models'))
inputs = sess.graph.get_tensor_by_name('input:0')
pred_cls_p = sess.graph.get_tensor_by_name('pred_cls:0')
pred_loc_p = sess.graph.get_tensor_by_name('pred_reg:0')

print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])) #number of parameters

with open('data1.txt') as f:
    for line in f.readlines():
        start = time.time()

        image = cv2.imread(line.split()[0])
        image = cv2.resize(image, (320, 240)) / 255.
        image_feed = np.expand_dims(image, 0)

        pred_cls, pred_reg = sess.run([pred_cls_p, pred_loc_p], feed_dict={inputs: image_feed})
        pred_cls, pred_reg = pred_cls[0], pred_reg[0]
        pred_cls = np.argmax(pred_cls, axis=1)
        indices = np.where(pred_cls > 0)[0]
        for i in indices:
            anchor = anchors[i]
            offset = pred_reg[i]
            cx = (anchor[0] + anchor[2]) / 2 + offset[0]
            cy = (anchor[1] + anchor[3]) / 2 + offset[1]
            w = (anchor[2] - anchor[0]) * np.exp(offset[2])
            h = (anchor[3] - anchor[1]) * np.exp(offset[3])
            p1 = (int(cx - w / 2), int(cy - h / 2))
            p2 = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(image, p1, p2, getColor(), 2)
        cv2.imshow('a', image)
        cv2.waitKey(3000)

        print(time.time()-start)
