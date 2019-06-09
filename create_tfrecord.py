import tensorflow as tf
from cfg import config
import numpy as np
import cv2


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def iou_mask(box, grid_yx, grid_anchors, iou_threshold=0.3):
    # @param
    # box: y1,x1,y2,x2 ,actual value
    # grid_yx : [center_y,center_x] , shape = [fm_h,fm_w,n_anchors,2]
    # grid_anchors: [h,w] shape = [1,1,n_anchors,2]
    # iou_threshold:if anchor and box iou>iou_threshold,this anchor could inference object
    offsets = np.zeros([config.fm_h, config.fm_w, config.num_anchors, 4],
                       dtype=np.float)  # last dimension offset:y,x,h,w
    box_cx = (box[1] + box[3]) / 2
    box_cy = (box[0] + box[2]) / 2
    box_h = box[2] - box[0]
    box_w = box[3] - box[1]
    # note grid_yx is the center_y and center_x of each cell,so at inference time we should use (top_cy+0.5,left_cy+0.5) as base_offset
    offsets[..., :2] += (np.array([[[[box_cy, box_cx]]]]) - grid_yx) * np.array([[[config.scale_factor]]])
    offsets[..., 2] += np.log(box_h / grid_anchors[..., 0])
    offsets[..., 3] += np.log(box_w / grid_anchors[..., 1])

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    grid_y1x1 = grid_yx - grid_anchors / 2  # top left
    grid_y2x2 = grid_yx + grid_anchors / 2  # bottom right

    y1 = np.maximum(box[0], grid_y1x1[..., 0])  # shape = [fm_h,fm_w,n_anchors]
    x1 = np.maximum(box[1], grid_y1x1[..., 1])
    y2 = np.minimum(box[2], grid_y2x2[..., 0])
    x2 = np.minimum(box[3], grid_y2x2[..., 1])

    h = np.maximum(0, y2 - y1)  # shape = [fm_h,fm_w,n_anchors]
    w = np.maximum(0, x2 - x1)

    intersection = h * w  # shape=[fm_h,fm_w,n_anchors]
    anchor_area = grid_anchors[..., 0] * grid_anchors[..., 1]  # shape = [1,1,n_anchors]
    union = box_area + anchor_area - intersection  # shape=[fm_h,fm_w,n_anchors]
    iou = intersection / union

    mask = iou > iou_threshold  # shape=[fm_h,fm_w,n_anchors]
    return mask, offsets


def get_mask(box):
    # box: y1,x1,y2,x2
    img_h, img_w, fm_h, fm_w = config.img_h, config.img_w, config.fm_h, config.fm_w

    grid_x = np.arange(fm_w, dtype=np.float)  # [0,1,2,...,w]
    grid_y = np.arange(fm_h, dtype=np.float)  # [0,1,2,...,h]
    # grid_x: shape=h*w                 grid_y: shape = h*w
    # [[0,1,2,...,w], //line 0          [[0,0,0,...,0], //line 0
    # [0,1,2,...,w], //line 1           [1,1,1,...,1], //line 1
    #   ...                               ....
    # [0,1,2,...,w]] //line h           [h,h,h,...,h]] //line h
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    grid_x, grid_y = np.expand_dims(grid_x, -1), np.expand_dims(grid_y, -1)
    grid_yx = np.concatenate([grid_y, grid_x], axis=-1)  # shape = [fm_h,fm_w,2]
    grid_yx = np.expand_dims(grid_yx, axis=2)  # [fm_h,m_w,1,2]
    grid_yx = np.repeat(grid_yx, config.num_anchors, axis=2)  # [m_h,fm_w,num_anchors,2]

    grid_yx += 0.5  # center_y,center_x in feature map
    grid_yx = grid_yx * [img_h / fm_h, img_w / fm_w]  # center_y,center_x in original image

    grid_anchors = np.array([[config.anchors]], dtype=np.float)  # shape = [1,1,num_anchors,2]

    mask, offsets = iou_mask(box, grid_yx, grid_anchors)
    return mask, offsets


def _parse(boxes):
    # boxes: box:x,y,w,h,c
    # return the true targets of objectness,locations,probabilities
    true_cls = np.zeros([config.fm_h, config.fm_w, config.num_anchors, config.num_classes], dtype=np.float)
    true_cls[..., 0] = 1  # setting  background
    true_loc = np.zeros([config.fm_h, config.fm_w, config.num_anchors, 4], dtype=np.float)
    for x, y, w, h, c in boxes:  # box in boxes : x,y,w,h,class
        mask, offsets = get_mask([y, x, y + h, x + w])
        true_cls[mask, c] = 1
        true_cls[mask, 0] = 0
        # TODO : if an anchor has two or more objects iou>iou_threshold,choose the biggest iou
        true_loc[mask] = offsets[mask]  # if mask_item is true,then true_loc_item = offsets_item
    return true_cls.flatten().tobytes(), true_loc.flatten().tobytes()


def createTFRecord():
    writer = tf.python_io.TFRecordWriter('train.tfrecords')
    with open('data1.txt') as f:
        for line in f.readlines():
            # line:imgpath x1 y1 w1 h1 c1 x2 y2 w2 h2 c2 ...
            line = line.split()
            imgpath = line[0]
            img = cv2.imread(imgpath)
            img = cv2.resize(img, (config.input_shape[2], config.input_shape[1]))
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
                boxes.append(newl[i * 5:i * 5 + 5])
            true_cls, true_loc = _parse(boxes)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': bytes_feature((img / 255.).flatten().tobytes()),
                'true_cls': bytes_feature(true_cls),
                'true_loc': bytes_feature(true_loc)
            }))
            writer.write(example.SerializeToString())
    writer.close()


def _parse2(example):
    features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'true_cls': tf.FixedLenFeature([], dtype=tf.string),
        'true_loc': tf.FixedLenFeature([], dtype=tf.string)
    }
    parsed_example = tf.parse_single_example(example, features)
    image = parsed_example['image']
    image = tf.decode_raw(image, out_type=tf.float64)
    image = tf.reshape(image, config.input_shape[1:])
    # image = tf.image.resize_images(image, [640,480])  # resize后为float类型
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    true_cls = parsed_example['true_cls']
    true_loc = parsed_example['true_loc']
    true_cls = tf.decode_raw(true_cls, out_type=tf.float64)
    true_cls = tf.reshape(true_cls, [config.fm_h, config.fm_w, config.num_anchors, config.num_classes])
    true_loc = tf.decode_raw(true_loc, out_type=tf.float64)
    true_loc = tf.reshape(true_loc, [config.fm_h, config.fm_w, config.num_anchors, 4])
    return image, true_cls, true_loc


def readTFReecord():
    dataset = tf.data.TFRecordDataset('train.tfrecords')
    dataset = dataset.map(_parse2).shuffle(buffer_size=1024).batch(16).repeat()
    it = dataset.make_one_shot_iterator()
    next_op = it.get_next()
    return next_op


def test():
    next_op = readTFReecord()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        images, true_cls, true_loc = sess.run(next_op)
        for i in range(16):
            image = images[i]
            loc = true_loc[i]
            cls = true_cls[i]  # [h,w,na,nc]
            cls[..., 0] = 0
            masks = np.array(np.where(cls > 0)).T
            for h1, w1, na1, nc1 in masks:
                py, px, ph, pw = loc[h1, w1, na1, :]

                tx = int((w1 + 0.5 + px) / config.feature_map_shape[1] * image.shape[1])
                ty = int((h1 + 0.5 + py) / config.feature_map_shape[0] * image.shape[0])
                w = int(config.anchors[na1][1] * np.exp(pw))
                h = int(config.anchors[na1][0] * np.exp(ph))
                print(tx, ty, w, h)
                cv2.rectangle(image, (tx - w // 2, ty - h // 2), (tx + w // 2, ty + h // 2), (0, 255, 0), 2)

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


if __name__ == '__main__':
    createTFRecord()
    test()
