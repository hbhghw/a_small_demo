from layers import *
from cfg import config
from create_tfrecord import readTFReecord


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def build_mobilenet_large(inputs,
                          width_multiplier: float = 1.0,
                          divisible_by: int = 8,
                          l2_reg: float = 1e-5, ):
    bneck_settings = [
    #    k   exp  out   SE       NL     s
        [3,  16,  16,  False,  "relu",  1],
        [3,  64,  24,  False,  "relu",  2],
        [3,  72,  24,  False,  "relu",  1],
        [5,  72,  40,  True,  "relu",   2],
        [5, 120,  40,  True,  "relu",   1],
        [5, 120,  40,  True,  "relu",   1],
        [3, 240,  80,  False, "hswish", 2],
        [3, 200,  80,  False, "hswish", 1],
        [3, 184,  80,  False, "hswish", 1],
        [3, 184,  80,  False, "hswish", 1],
        [3, 480, 112,  True, "hswish",  1],
        [3, 672, 112,  True, "hswish",  1],
        [5, 672, 160,  True, "hswish",  2],
        [5, 960, 160,  True, "hswish",  1],
        [5, 960, 160,  True, "hswish",  1]
    ]
    x = conv_bn_relu(inputs,
                     16,
                     kernel_size=3,
                     stride=2,
                     padding=1,
                     norm_layer="bn",
                     act_layer="hswish",
                     use_bias=False,
                     l2_reg=l2_reg
                     )

    for idx, (k, exp, out, SE, NL, s) in enumerate(bneck_settings):
        out_channels = _make_divisible(out * width_multiplier, divisible_by)
        exp_channels = _make_divisible(exp * width_multiplier, divisible_by)
        x = bneck(x,
                  out_channels=out_channels,
                  exp_channels=exp_channels,
                  kernel_size=k,
                  stride=s,
                  use_se=SE,
                  act_layer=NL, index=idx
                  )
    ##[None,8,10,160]
    pred_cls = tf.layers.conv2d(x, config.num_anchors * config.num_classes, [3, 3], 1, padding='same')
    pred_cls = tf.reshape(pred_cls, [-1, config.fm_h, config.fm_w, config.num_anchors, config.num_classes],
                          name='pred_cls')
    pred_cls = tf.nn.softmax(pred_cls, axis=-1)

    pred_loc = tf.layers.conv2d(x, config.num_anchors * 4, [3, 3], 1, padding='same')
    pred_loc = tf.reshape(pred_loc, [-1, config.fm_h, config.fm_w, config.num_anchors, 4],
                          name='pred_loc')  # the last dimention is x,y,pw,ph

    return pred_cls, pred_loc


def build_model():
    # ----------------------- build model ----------------------
    inputs = tf.placeholder(dtype=tf.float32, shape=config.input_shape, name='input')
    true_cls = tf.placeholder(dtype=tf.float64,
                              shape=[None, config.fm_h, config.fm_w, config.num_anchors, config.num_classes],
                              name='true_cls')
    true_loc = tf.placeholder(dtype=tf.float64, shape=[None, config.fm_h, config.fm_w, config.num_anchors, 4],
                              name='true_loc')
    true_cls = tf.stop_gradient(true_cls)
    true_loc = tf.stop_gradient(true_loc)

    true_cls, true_loc = tf.cast(true_cls, tf.float32), tf.cast(true_loc, tf.float32)

    pred_cls, pred_loc = build_mobilenet_large(inputs)

    # ------------------- classify loss -------------------------

    cls_loss_all = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_cls, logits=pred_cls,
                                                              axis=-1)  # [-1,h,w,n_anchors]
    neg_mask = true_cls[..., 0]  # [-1,h,w,n_anchors] , background mask
    # pos_mask = true_cls[...,1:] #shape=[-1,h,w,na,nc-1]
    pos_mask = 1 - neg_mask  #shape = [-1,h,w,n_anchors], if not bg,then must be an obj

    def cls_loss1():
        # calculate all loss with weights
        n_neg = tf.reduce_sum(neg_mask)
        n_pos = tf.reduce_sum(pos_mask)
        n_pos = tf.where(n_pos > 0, n_pos, 1.0)  # incase n_pos==0
        cls_neg_loss = config.neg_pos_ratio / n_neg * (neg_mask * cls_loss_all)
        cls_pos_loss = 1 / n_pos * (pos_mask * cls_loss_all)
        return tf.reduce_sum(cls_neg_loss) + tf.reduce_sum(cls_pos_loss)

    def cls_loss2():
        # choose top_k neg loss
        n_neg = tf.to_int32(tf.reduce_sum(neg_mask))  # int32
        n_pos = tf.to_int32(tf.reduce_sum(pos_mask))
        no_obj_all = neg_mask * cls_loss_all  # [-1,h,w,n_anchors]

        n_neg_keep = tf.minimum(tf.maximum(config.neg_pos_ratio * n_pos, config.n_neg_min), n_neg)
        no_obj_all_1D = tf.reshape(no_obj_all, [-1])
        values, indices = tf.nn.top_k(no_obj_all_1D, k=n_neg_keep, sorted=False)
        neg_keep_mask = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                      updates=tf.ones_like(indices, dtype=tf.int32),
                                      shape=tf.shape(no_obj_all_1D))
        neg_keep_mask = tf.to_float(tf.reshape(neg_keep_mask, [-1, config.fm_h, config.fm_w, config.num_anchors]))
        no_obj_loss = tf.reduce_sum(no_obj_all * neg_keep_mask)

        obj_loss = tf.reduce_sum(pos_mask * cls_loss_all)
        return no_obj_loss + obj_loss

    cls_loss = cls_loss1()
    # ------------------- classify loss end --------------------------

    # -------------------regression loss -----------------------------
    pred_loc_xy = tf.nn.sigmoid(pred_loc[..., :2])  # [-1,h,w,na,2]
    pred_loc_wh = pred_loc[..., 2:]  # [-1,h,w,na,2]

    def smooth_L1(x):
        l1 = tf.abs(x) - 0.5
        l2 = 0.5 * tf.square(x)
        mask = tf.abs(x) < 1.0
        x = tf.where(mask, l2, l1)
        return x

    # note : location regression loss,we only compute the pos objects,
    loc_loss_xy_all = tf.reduce_sum(smooth_L1(pred_loc_xy - true_loc[..., :2]),axis=-1) #[-1,h,w,na]
    loc_loss_wh_all = tf.reduce_sum(smooth_L1(pred_loc_wh - true_loc[..., 2:]),axis=-1) #[-1,h,w,na]

    loc_loss_xy = tf.reduce_sum(loc_loss_xy_all*pos_mask)
    loc_loss_wh = tf.reduce_sum(loc_loss_wh_all*pos_mask)
    loc_loss = loc_loss_xy + loc_loss_wh
    # ------------------regression loss end --------------------------

    losses = cls_loss + loc_loss

    tf.summary.scalar('cls_loss', cls_loss)
    tf.summary.scalar('loc_loss_xy', loc_loss_xy)
    tf.summary.scalar('loc_loss_wh', loc_loss_wh)
    tf.summary.scalar('loc_loss', loc_loss)
    tf.summary.scalar('losses', losses)
    summary_op = tf.summary.merge_all()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(losses)

    return inputs, true_cls, true_loc, losses, train_op, summary_op


def train():
    next_op = readTFReecord()

    inputs, cls_placeholder, loc_placeholder, losses, train_op, summary_op = build_model()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('logs')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1001):
            images, true_cls, true_loc = sess.run(next_op)
            images = images / 255.  # int to float
            _, loss, summary = sess.run([train_op, losses, summary_op],
                                        feed_dict={inputs: images, cls_placeholder: true_cls,
                                                   loc_placeholder: true_loc})
            summary_writer.add_summary(summary, i)
            if i % 50 == 0:
                print('step', i, 'loss:', loss)
                saver.save(sess, 'models/model', global_step=i)


if __name__ == '__main__':
    train()
