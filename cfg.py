from easydict import EasyDict

config = EasyDict()

config.input_shape = [None, 240, 320, 3]
config.img_h = config.input_shape[1]
config.img_w = config.input_shape[2]

config.feature_map_shape = [8, 10]
config.fm_h = config.feature_map_shape[0]
config.fm_w = config.feature_map_shape[1]
config.scale_factor = [config.fm_h / config.img_h, config.fm_w / config.img_w]

config.anchors = [[20, 20], [40, 30], [50, 42], [50, 60], [75, 45], [90, 60],
                  [110, 80]]  # [h,w] need to be carefully designed
config.num_anchors = len(config.anchors)
config.num_classes = 2

config.neg_pos_ratio = 3
config.n_neg_min = 5
