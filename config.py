from easydict import EasyDict

cfg = EasyDict()

cfg.input_shape = [None, 240, 320, 3]
cfg.img_h = cfg.input_shape[1]
cfg.img_w = cfg.input_shape[2]

cfg.feature_maps = [
    [30, 40],
    [15, 20],
    [8, 10],
    [4, 5]
]

cfg.anchors = [
    [[10, 10], [20, 15],[35,21]],
    [[30, 25], [50, 35],[65,45]],
    [[50, 35], [65, 50]],
    [[60, 45], [100,80]]
]  # [h,w] need to be carefully designed
cfg.num_anchors = sum([cfg.feature_maps[i][0] * cfg.feature_maps[i][1] * len(cfg.anchors[i]) for i in
                       range(len(cfg.feature_maps))])
cfg.num_classes = 2

cfg.neg_pos_ratio = 3
cfg.n_neg_min = 5

cfg.batch_size = 16
