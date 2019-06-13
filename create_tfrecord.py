import tensorflow as tf
from config import cfg
import numpy as np
import cv2


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def compute_iou(box, anchor):
    minx = max(box[0], anchor[0])
    miny = max(box[1], anchor[1])
    maxx = min(box[2], anchor[2])
    maxy = min(box[3], anchor[3])
    w = max(maxx - minx, 0)
    h = max(maxy - miny, 0)
    inter = w * h
    union = (box[2] - box[0]) * (box[3] - box[1]) + (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]) - inter
    return inter / union


def compute_offset(box, anchor):
    dx = (box[2] + box[0] - anchor[2] - anchor[0]) / 2
    dy = (box[3] + box[1] - anchor[3] - anchor[1]) / 2
    dw = np.log((box[2] - box[0]) / (anchor[2] - anchor[0]))
    dh = np.log((box[3] - box[1]) / (anchor[3] - anchor[1]))
    return dx, dy, dw, dh


def get_anchors():
    '''

    :return: all anchors x1,y1,x2,y2
    '''
    anchors = []
    for fm, fm_anchors in zip(cfg.feature_maps, cfg.anchors):
        tmp = np.zeros([fm[0], fm[1], len(fm_anchors), 4], dtype=np.float)
        y = np.arange(fm[0], dtype=np.float)
        x = np.arange(fm[1], dtype=np.float)
        x, y = np.meshgrid(x, y)
        x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
        yx = np.concatenate([y, x], axis=-1)
        yx = yx + 0.5
        down_scale = [fm[0] / cfg.img_h, fm[1] / cfg.img_w]
        yx = yx / down_scale
        yx = np.expand_dims(yx, axis=2)  # [h,w,1,2]
        yx = np.repeat(yx, repeats=len(fm_anchors), axis=2)
        fm_anchors = np.reshape(np.array(fm_anchors), [1, 1, -1, 2])
        tmp[..., :2] += (yx - fm_anchors / 2)
        tmp[..., 2:] += (yx + fm_anchors / 2)
        anchors.append(np.reshape(tmp, [-1, 4]))
    anchors = np.concatenate(anchors, axis=0) #notice:y1,x1,y2,x2 format!
    anchors = np.vstack((anchors[:,1],anchors[:,0],anchors[:,3],anchors[:,2])).T #change to x1,y1,x2,y2 format
    return anchors  # [-1,4]


def get_true_label(boxes, anchors, iou_threshold=0.5):
    # boxes and anchors : x1,y1,x2,y2
    true_cls = np.zeros([cfg.num_anchors, 2]);
    true_cls[:, 0] = 1.
    true_reg = np.zeros([cfg.num_anchors, 4])
    tmp_iou = np.zeros([cfg.num_anchors])
    for box in boxes:
        for i in range(cfg.num_anchors):
            anchor = anchors[i]
            iou = compute_iou(box, anchor)
            if iou > iou_threshold:
                true_cls[i] = [0, 1]
                if iou > tmp_iou[i]:
                    tmp_iou[i] = iou
                    true_reg[i] = compute_offset(box, anchor)
    return true_cls, true_reg


def createTFRecord():
    anchors = get_anchors()
    writer = tf.python_io.TFRecordWriter('train.tfrecords')
    with open('data1.txt') as f:
        for line in f.readlines():
            # line:imgpath x1 y1 w1 h1 c1 x2 y2 w2 h2 c2 ...
            line = line.split()
            imgpath = line[0]
            img = cv2.imread(imgpath)
            img = cv2.resize(img, (cfg.input_shape[2], cfg.input_shape[1]))
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # if resized ,x,y,w,h also need be scaled except c
            newl = []
            for i in range(1, len(line)):
                if i % 5:
                    newl.append(int(line[i]) / 2)  # resize scale = 2
                else:
                    newl.append(int(line[i]))  # labels won't change
            boxes = []
            for i in range(len(newl) // 5):
                boxes.append(
                    [newl[i * 5], newl[i * 5 + 1], newl[i * 5] + newl[i * 5 + 2], newl[i * 5 + 1] + newl[i * 5 + 3]])
            true_cls, true_reg = get_true_label(boxes, anchors)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': bytes_feature((img / 255.).flatten().tobytes()),
                'true_cls': bytes_feature(true_cls.flatten().tobytes()),
                'true_reg': bytes_feature(true_reg.flatten().tobytes())
            }))
            writer.write(example.SerializeToString())
    writer.close()


def _parse(example):
    features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'true_cls': tf.FixedLenFeature([], dtype=tf.string),
        'true_reg': tf.FixedLenFeature([], dtype=tf.string)
    }
    parsed_example = tf.parse_single_example(example, features)
    image = parsed_example['image']
    image = tf.decode_raw(image, out_type=tf.float64)
    image = tf.reshape(image, cfg.input_shape[1:])
    # image = tf.image.resize_images(image, [640,480])  # resize后为float类型
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    true_cls = parsed_example['true_cls']
    true_reg = parsed_example['true_reg']
    true_cls = tf.decode_raw(true_cls, out_type=tf.float64)
    true_cls = tf.reshape(true_cls, [cfg.num_anchors, 2])
    true_reg = tf.decode_raw(true_reg, out_type=tf.float64)
    true_reg = tf.reshape(true_reg, [cfg.num_anchors, 4])
    return image, true_cls, true_reg


def readTFReecord():
    dataset = tf.data.TFRecordDataset('train.tfrecords')
    dataset = dataset.map(_parse).shuffle(buffer_size=1024).batch(cfg.batch_size).repeat()
    it = dataset.make_one_shot_iterator()
    next_op = it.get_next()
    return next_op


def getColor():
    color = [0,0,0]
    for i in range(3):
        c = np.random.random()
        color[i] = c
    return color

def test():
    next_op = readTFReecord()
    anchors = get_anchors()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        images, true_cls, true_reg = sess.run(next_op)
        for i in range(cfg.batch_size):
            image = images[i]
            loc = true_reg[i]
            cls = true_cls[i]  # [h,w,na,nc]
            indices = np.where(cls[:, 1] > 0)[0]
            for index in indices:
                anchor = anchors[index]
                cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), getColor(), 1) #show matched anchors
                # dx, dy, dw, dh = loc[index]
                # cx = (anchor[2] + anchor[0]) / 2 + dx
                # cy = (anchor[3] + anchor[1]) / 2 + dy
                # tw = (anchor[2] - anchor[0]) * np.exp(dw)
                # th = (anchor[3] - anchor[1]) * np.exp(dh)
                # p1 = (int(cx - tw / 2), int(cy - th / 2))
                # p2 = (int(cx + tw / 2), int(cy + th / 2))
                # cv2.rectangle(image, p1, p2, (0, 255, 0), 2)


            cv2.imshow("a", image)
            cv2.waitKey(1000)


def showFigure():
    import matplotlib.pyplot as plt

    w = []
    h = []
    with open('data1.txt') as f:
        for l in f.readlines():
            _w, _h = l.strip().split()[-3:-1]
            w.append(int(_w) / 2)  # div 2 since image resize scale=2
            h.append(int(_h) / 2)
    print(min(w), min(h), max(w), max(h))
    plt.scatter(h, w)
    plt.show()

def show_anchors():
    anchors = get_anchors()
    image = np.zeros([240,320,3])
    for anchor in anchors:
        cv2.rectangle(image, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), getColor(), 1)
    cv2.imshow('a',image)
    cv2.waitKey(10000)

if __name__ == '__main__':
    createTFRecord()
    test()
    # show_anchors()